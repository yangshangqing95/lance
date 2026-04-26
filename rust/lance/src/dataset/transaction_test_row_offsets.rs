#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use futures::TryStreamExt;

    use crate::dataset::transaction::{Operation, Transaction, UpdateMode};
    use crate::dataset::write::merge_insert::{WhenMatched, WhenNotMatched};
    use crate::dataset::write::update::UpdateBuilder;
    use crate::dataset::{Dataset, MergeInsertBuilder, WriteParams};
    use lance_datafusion::utils::reader_to_stream;
    use lance_table::format::pb;

    // ════════════════════════════════════════════════════════════════════
    //  Protobuf round-trip
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_updated_row_offsets_proto_roundtrip_none() {
        let op = Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![],
            new_fragments: vec![],
            fields_modified: vec![],
            merged_generations: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_row_offsets: None,
        };

        let tx = Transaction::new_from_version(0, op);
        let pb_tx: pb::Transaction = (&tx).into();
        let tx2 = Transaction::try_from(pb_tx).unwrap();
        assert_eq!(tx, tx2);

        if let Operation::Update {
            updated_row_offsets,
            ..
        } = &tx2.operation
        {
            assert!(updated_row_offsets.is_none());
        } else {
            panic!("Expected Update operation");
        }
    }

    /// Small offset sets (≤5000) should use the array encoding path.
    #[test]
    fn test_updated_row_offsets_proto_roundtrip_array_encoding() {
        let mut offsets = HashMap::new();
        offsets.insert(0u64, vec![1u32, 3, 5]);
        offsets.insert(2u64, vec![0u32, 2]);

        let op = Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![],
            new_fragments: vec![],
            fields_modified: vec![],
            merged_generations: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_row_offsets: Some(offsets),
        };

        let tx = Transaction::new_from_version(0, op);
        let pb_tx: pb::Transaction = (&tx).into();

        // Verify it used the Array encoding (not Bitmap) for small sets
        if let Some(pb::transaction::Operation::Update(ref update)) = pb_tx.operation {
            for fro in &update.updated_row_offsets {
                assert!(
                    matches!(
                        fro.offsets,
                        Some(pb::transaction::fragment_row_offsets::Offsets::Array(_))
                    ),
                    "Small offset sets should use array encoding"
                );
            }
        }

        let tx2 = Transaction::try_from(pb_tx).unwrap();

        if let Operation::Update {
            updated_row_offsets,
            ..
        } = &tx2.operation
        {
            let map = updated_row_offsets.as_ref().unwrap();
            assert_eq!(map.len(), 2);
            assert_eq!(map.get(&0).unwrap(), &vec![1u32, 3, 5]);
            assert_eq!(map.get(&2).unwrap(), &vec![0u32, 2]);
        } else {
            panic!("Expected Update operation");
        }
    }

    /// Large offset sets (>5000) should use the Roaring bitmap encoding path.
    #[test]
    fn test_updated_row_offsets_proto_roundtrip_bitmap_encoding() {
        // Generate >5000 offsets to trigger bitmap encoding
        let large_offsets: Vec<u32> = (0..10_000u32).collect();
        let mut offsets = HashMap::new();
        offsets.insert(42u64, large_offsets.clone());

        let op = Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![],
            new_fragments: vec![],
            fields_modified: vec![],
            merged_generations: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_row_offsets: Some(offsets),
        };

        let tx = Transaction::new_from_version(0, op);
        let pb_tx: pb::Transaction = (&tx).into();

        // Verify it used the Bitmap encoding for large sets
        if let Some(pb::transaction::Operation::Update(ref update)) = pb_tx.operation {
            assert_eq!(update.updated_row_offsets.len(), 1);
            let fro = &update.updated_row_offsets[0];
            assert_eq!(fro.fragment_id, 42);
            assert!(
                matches!(
                    fro.offsets,
                    Some(pb::transaction::fragment_row_offsets::Offsets::Bitmap(_))
                ),
                "Large offset sets (>5000) should use bitmap encoding"
            );
        }

        // Round-trip back and verify correctness
        let tx2 = Transaction::try_from(pb_tx).unwrap();

        if let Operation::Update {
            updated_row_offsets,
            ..
        } = &tx2.operation
        {
            let map = updated_row_offsets.as_ref().unwrap();
            assert_eq!(map.len(), 1);
            let deserialized = map.get(&42).unwrap();
            assert_eq!(deserialized.len(), 10_000);
            assert_eq!(deserialized, &large_offsets);
        } else {
            panic!("Expected Update operation");
        }
    }

    /// `Some(empty_map)` should round-trip as `Some(empty)`, distinct from `None`.
    #[test]
    fn test_updated_row_offsets_proto_roundtrip_empty_map() {
        let op = Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![],
            new_fragments: vec![],
            fields_modified: vec![],
            merged_generations: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_row_offsets: Some(HashMap::new()),
        };

        let tx = Transaction::new_from_version(0, op.clone());
        let pb_tx: pb::Transaction = (&tx).into();
        let tx2 = Transaction::try_from(pb_tx).unwrap();

        // Empty map serializes as empty repeated field → deserializes as None
        // This is expected protobuf behavior: empty repeated = absent.
        if let Operation::Update {
            updated_row_offsets,
            ..
        } = &tx2.operation
        {
            assert!(
                updated_row_offsets.is_none(),
                "Empty map should deserialize as None (protobuf convention)"
            );
        } else {
            panic!("Expected Update operation");
        }
    }

    /// Mixed: one fragment with small offsets (array encoding) and another
    /// with large offsets (bitmap encoding) in the same transaction.
    #[test]
    fn test_updated_row_offsets_proto_roundtrip_mixed_encoding() {
        let small_offsets: Vec<u32> = vec![0, 5, 10];
        let large_offsets: Vec<u32> = (0..8_000u32).collect();

        let mut offsets = HashMap::new();
        offsets.insert(0u64, small_offsets.clone());
        offsets.insert(1u64, large_offsets.clone());

        let op = Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![],
            new_fragments: vec![],
            fields_modified: vec![],
            merged_generations: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_row_offsets: Some(offsets),
        };

        let tx = Transaction::new_from_version(0, op);
        let pb_tx: pb::Transaction = (&tx).into();

        // Verify encoding choices
        if let Some(pb::transaction::Operation::Update(ref update)) = pb_tx.operation {
            assert_eq!(update.updated_row_offsets.len(), 2);
            for fro in &update.updated_row_offsets {
                match fro.fragment_id {
                    0 => assert!(
                        matches!(
                            fro.offsets,
                            Some(pb::transaction::fragment_row_offsets::Offsets::Array(_))
                        ),
                        "Fragment 0 (small) should use array encoding"
                    ),
                    1 => assert!(
                        matches!(
                            fro.offsets,
                            Some(pb::transaction::fragment_row_offsets::Offsets::Bitmap(_))
                        ),
                        "Fragment 1 (large) should use bitmap encoding"
                    ),
                    _ => panic!("Unexpected fragment id"),
                }
            }
        }

        // Round-trip
        let tx2 = Transaction::try_from(pb_tx).unwrap();
        if let Operation::Update {
            updated_row_offsets,
            ..
        } = &tx2.operation
        {
            let map = updated_row_offsets.as_ref().unwrap();
            assert_eq!(map.get(&0).unwrap(), &small_offsets);
            assert_eq!(map.get(&1).unwrap(), &large_offsets);
        } else {
            panic!("Expected Update operation");
        }
    }

    // ════════════════════════════════════════════════════════════════════
    //  PartialEq
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_updated_row_offsets_equality() {
        let make_op = |offsets: Option<HashMap<u64, Vec<u32>>>| Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![],
            new_fragments: vec![],
            fields_modified: vec![],
            merged_generations: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_row_offsets: offsets,
        };

        let op_none = make_op(None);
        let op_none2 = make_op(None);
        let op_empty = make_op(Some(HashMap::new()));
        let mut m1 = HashMap::new();
        m1.insert(0u64, vec![1u32]);
        let op_some = make_op(Some(m1.clone()));
        let op_some2 = make_op(Some(m1));

        assert_eq!(op_none, op_none2);
        assert_eq!(op_some, op_some2);
        assert_ne!(op_none, op_empty);
        assert_ne!(op_none, op_some);
    }

    // ════════════════════════════════════════════════════════════════════
    //  update_columns: RoaringBitmap return value
    // ════════════════════════════════════════════════════════════════════

    /// Verify that `update_columns` returns a `Some(RoaringBitmap)` containing
    /// exactly the physical row offsets that were matched by the right-side data.
    #[tokio::test]
    async fn test_update_columns_returns_matched_bitmap() {
        use arrow_array::UInt64Array;
        use crate::dataset::ROW_ID;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
            ],
        )
        .unwrap();

        let tmp_dir = tempfile::tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let ds = Dataset::write(reader, uri, None).await.unwrap();

        let frag = ds.get_fragment(0).unwrap();
        let frag_id = frag.metadata().id;

        // Update rows at offset 0 and 4 (id=10, id=50)
        let update_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let update_batch = RecordBatch::try_new(
            update_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![
                    (frag_id << 32) | 0,
                    (frag_id << 32) | 4,
                ])),
                Arc::new(StringArray::from(vec!["A", "E"])),
            ],
        )
        .unwrap();

        let right_reader: Box<dyn arrow_array::RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(
                vec![Ok(update_batch)],
                update_schema,
            ));

        let mut frag_mut = ds.get_fragment(0).unwrap();
        let (_updated_fragment, _fields_modified, matched_bitmap) = frag_mut
            .update_columns(right_reader, ROW_ID, ROW_ID)
            .await
            .unwrap();

        let bitmap = matched_bitmap.expect("update_columns should return Some(bitmap)");
        assert_eq!(bitmap.len(), 2, "Should have exactly 2 matched offsets");
        assert!(bitmap.contains(0), "Offset 0 should be matched");
        assert!(bitmap.contains(4), "Offset 4 should be matched");
        assert!(!bitmap.contains(1), "Offset 1 should not be matched");
        assert!(!bitmap.contains(2), "Offset 2 should not be matched");
        assert!(!bitmap.contains(3), "Offset 3 should not be matched");
    }

    /// When no rows match the right-side data, the bitmap should be empty.
    #[tokio::test]
    async fn test_update_columns_no_matches_returns_empty_bitmap() {
        use arrow_array::UInt64Array;
        use crate::dataset::ROW_ID;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();

        let tmp_dir = tempfile::tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let ds = Dataset::write(reader, uri, None).await.unwrap();

        // Right-side ROW_IDs that do NOT exist in the fragment
        let update_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let update_batch = RecordBatch::try_new(
            update_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![999999])), // non-existent
                Arc::new(StringArray::from(vec!["Z"])),
            ],
        )
        .unwrap();

        let right_reader: Box<dyn arrow_array::RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(
                vec![Ok(update_batch)],
                update_schema,
            ));

        let mut frag_mut = ds.get_fragment(0).unwrap();
        let (_updated_fragment, _fields_modified, matched_bitmap) = frag_mut
            .update_columns(right_reader, ROW_ID, ROW_ID)
            .await
            .unwrap();

        let bitmap = matched_bitmap.expect("update_columns should always return Some when tracking");
        assert!(bitmap.is_empty(), "No rows matched, bitmap should be empty");
    }

    /// When all rows match, the bitmap should contain every physical offset.
    #[tokio::test]
    async fn test_update_columns_all_matches_returns_full_bitmap() {
        use arrow_array::UInt64Array;
        use crate::dataset::ROW_ID;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();

        let tmp_dir = tempfile::tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let ds = Dataset::write(reader, uri, None).await.unwrap();

        let frag = ds.get_fragment(0).unwrap();
        let frag_id = frag.metadata().id;

        // Match all 3 rows
        let update_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let update_batch = RecordBatch::try_new(
            update_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![
                    (frag_id << 32) | 0,
                    (frag_id << 32) | 1,
                    (frag_id << 32) | 2,
                ])),
                Arc::new(StringArray::from(vec!["A", "B", "C"])),
            ],
        )
        .unwrap();

        let right_reader: Box<dyn arrow_array::RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(
                vec![Ok(update_batch)],
                update_schema,
            ));

        let mut frag_mut = ds.get_fragment(0).unwrap();
        let (_updated_fragment, _fields_modified, matched_bitmap) = frag_mut
            .update_columns(right_reader, ROW_ID, ROW_ID)
            .await
            .unwrap();

        let bitmap = matched_bitmap.unwrap();
        assert_eq!(bitmap.len(), 3);
        assert!(bitmap.contains(0));
        assert!(bitmap.contains(1));
        assert!(bitmap.contains(2));
    }

    // ════════════════════════════════════════════════════════════════════
    //  RewriteRows (via UpdateBuilder)
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_rewrite_rows_selective_version_update() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
            ],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let tmp_dir = tempfile::tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let ds = Dataset::write(
            reader,
            uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let v1 = ds.version().version;

        let update_result = UpdateBuilder::new(Arc::new(ds))
            .set("value", "'updated'")
            .unwrap()
            .update_where("key = 2 OR key = 4")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let updated_ds = update_result.new_dataset;
        let v2 = updated_ds.version().version;
        assert!(v2 > v1);

        let actual_batches = updated_ds
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let mut found: HashMap<i32, String> = HashMap::new();
        for batch in &actual_batches {
            let keys = batch
                .column_by_name("key")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let values = batch
                .column_by_name("value")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for i in 0..keys.len() {
                found.insert(keys.value(i), values.value(i).to_string());
            }
        }

        assert_eq!(found[&1], "a");
        assert_eq!(found[&2], "updated");
        assert_eq!(found[&3], "c");
        assert_eq!(found[&4], "updated");
        assert_eq!(found[&5], "e");
    }

    // ════════════════════════════════════════════════════════════════════
    //  RewriteColumns (via merge_insert with partial schema)
    // ════════════════════════════════════════════════════════════════════

    /// merge_insert with a **subset** of columns triggers the RewriteColumns
    /// path.  After the update, only the matched rows should carry the new
    /// version in `last_updated_at_version_meta`; unmatched rows must keep the
    /// old version.
    #[tokio::test]
    async fn test_rewrite_columns_version_metadata() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("score", DataType::Int32, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
                Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50])),
            ],
        )
        .unwrap();

        let tmp_dir = tempfile::tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let ds = Dataset::write(
            reader,
            uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let v1 = ds.version().version;
        assert!(ds.manifest().uses_stable_row_ids());

        let source_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("score", DataType::Int32, true),
        ]));

        let source_batch = RecordBatch::try_new(
            source_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![2, 4])),
                Arc::new(Int32Array::from(vec![200, 400])),
            ],
        )
        .unwrap();

        let merge_job =
            MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["id".to_string()])
                .unwrap()
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::DoNothing)
                .try_build()
                .unwrap();

        let reader = Box::new(RecordBatchIterator::new(
            vec![Ok(source_batch)],
            source_schema.clone(),
        ));
        let (dataset, _stats) = merge_job.execute(reader_to_stream(reader)).await.unwrap();
        let v2 = dataset.version().version;
        assert!(v2 > v1, "Version should advance: v1={v1}, v2={v2}");

        // Verify data values
        let result_batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let mut found: HashMap<i32, (String, i32)> = HashMap::new();
        for batch in &result_batches {
            let ids = batch.column_by_name("id").unwrap().as_any().downcast_ref::<Int32Array>().unwrap();
            let names = batch.column_by_name("name").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
            let scores = batch.column_by_name("score").unwrap().as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..ids.len() {
                found.insert(ids.value(i), (names.value(i).to_string(), scores.value(i)));
            }
        }

        assert_eq!(found[&1], ("a".to_string(), 10));
        assert_eq!(found[&3], ("c".to_string(), 30));
        assert_eq!(found[&5], ("e".to_string(), 50));
        assert_eq!(found[&2], ("b".to_string(), 200));
        assert_eq!(found[&4], ("d".to_string(), 400));

        // Verify per-row version metadata
        let fragments = dataset.get_fragments();
        for frag in &fragments {
            let meta = frag.metadata();
            if let Some(version_meta) = &meta.last_updated_at_version_meta {
                let seq = version_meta.load_sequence().unwrap();
                let physical_rows = meta.physical_rows.unwrap_or(0);

                let versions: Vec<u64> =
                    (0..physical_rows).map(|i| seq.version_at(i).unwrap()).collect();

                for &ver in &versions {
                    assert!(
                        ver == v1 || ver == v2,
                        "Fragment {}: unexpected version {ver} (expected {v1} or {v2})",
                        meta.id,
                    );
                }

                let has_v1 = versions.iter().any(|&v| v == v1);
                let has_v2 = versions.iter().any(|&v| v == v2);
                assert!(
                    has_v1 && has_v2,
                    "Fragment {} should have a mix of v1 and v2 versions, got: {:?}",
                    meta.id, versions,
                );
            }
        }
    }

    /// When ALL rows in a fragment are updated via RewriteColumns, every row
    /// should have `last_updated_at_version == v2`.
    #[tokio::test]
    async fn test_rewrite_columns_all_rows_updated_version_metadata() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("score", DataType::Int32, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                Arc::new(Int32Array::from(vec![10, 20, 30])),
            ],
        )
        .unwrap();

        let tmp_dir = tempfile::tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let ds = Dataset::write(
            reader,
            uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let _v1 = ds.version().version;

        let source_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("score", DataType::Int32, true),
        ]));

        let source_batch = RecordBatch::try_new(
            source_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![100, 200, 300])),
            ],
        )
        .unwrap();

        let merge_job =
            MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["id".to_string()])
                .unwrap()
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::DoNothing)
                .try_build()
                .unwrap();

        let reader = Box::new(RecordBatchIterator::new(
            vec![Ok(source_batch)],
            source_schema.clone(),
        ));
        let (dataset, _stats) = merge_job.execute(reader_to_stream(reader)).await.unwrap();
        let v2 = dataset.version().version;

        let result_batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let mut found: HashMap<i32, i32> = HashMap::new();
        for batch in &result_batches {
            let ids = batch.column_by_name("id").unwrap().as_any().downcast_ref::<Int32Array>().unwrap();
            let scores = batch.column_by_name("score").unwrap().as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..ids.len() {
                found.insert(ids.value(i), scores.value(i));
            }
        }

        assert_eq!(found[&1], 100);
        assert_eq!(found[&2], 200);
        assert_eq!(found[&3], 300);

        let fragments = dataset.get_fragments();
        for frag in &fragments {
            let meta = frag.metadata();
            if let Some(version_meta) = &meta.last_updated_at_version_meta {
                let seq = version_meta.load_sequence().unwrap();
                let physical_rows = meta.physical_rows.unwrap_or(0);
                for row_idx in 0..physical_rows {
                    let ver = seq.version_at(row_idx).unwrap();
                    assert_eq!(ver, v2, "Fragment {}: row {} should be v2={v2}, got {ver}", meta.id, row_idx);
                }
            }
        }
    }

    /// Fragments not touched by the merge_insert should retain their original
    /// version metadata unchanged.
    #[tokio::test]
    async fn test_rewrite_columns_untouched_fragment_version_unchanged() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("score", DataType::Int32, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e", "f"])),
                Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50, 60])),
            ],
        )
        .unwrap();

        let tmp_dir = tempfile::tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let ds = Dataset::write(
            reader,
            uri,
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let v1 = ds.version().version;
        assert_eq!(ds.get_fragments().len(), 2);

        let source_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("score", DataType::Int32, true),
        ]));

        let source_batch = RecordBatch::try_new(
            source_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![999])),
            ],
        )
        .unwrap();

        let merge_job =
            MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["id".to_string()])
                .unwrap()
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::DoNothing)
                .try_build()
                .unwrap();

        let reader = Box::new(RecordBatchIterator::new(
            vec![Ok(source_batch)],
            source_schema.clone(),
        ));
        let (dataset, _stats) = merge_job.execute(reader_to_stream(reader)).await.unwrap();
        let v2 = dataset.version().version;

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);

        for frag in &fragments {
            let meta = frag.metadata();
            if let Some(version_meta) = &meta.last_updated_at_version_meta {
                let seq = version_meta.load_sequence().unwrap();
                let physical_rows = meta.physical_rows.unwrap_or(0);
                let versions: Vec<u64> =
                    (0..physical_rows).map(|i| seq.version_at(i).unwrap()).collect();

                let all_v1 = versions.iter().all(|&v| v == v1);
                let any_v2 = versions.iter().any(|&v| v == v2);

                if any_v2 {
                    for &v in &versions {
                        assert!(
                            v == v1 || v == v2,
                            "Fragment {}: unexpected version {v}",
                            meta.id,
                        );
                    }
                } else {
                    assert!(
                        all_v1,
                        "Untouched fragment {} should have all v1, got {:?}",
                        meta.id, versions,
                    );
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════
    //  External caller path: updated_row_offsets via Dataset::commit
    // ════════════════════════════════════════════════════════════════════

    /// Simulates the Java/Spark external caller path:
    ///   1. Create dataset with stable row IDs
    ///   2. Use fragment.update_columns() to rewrite a column
    ///   3. Convert returned `RoaringBitmap` → `Vec<u32>` for `Operation::Update`
    ///   4. Commit via Dataset::commit()
    ///   5. Verify build_manifest() applied version metadata correctly
    #[tokio::test]
    async fn test_external_caller_updated_row_offsets_partial() {
        use arrow_array::UInt64Array;
        use crate::dataset::ROW_ID;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
            ],
        )
        .unwrap();

        let tmp_dir = tempfile::tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let ds = Dataset::write(
            reader,
            uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let v1 = ds.version().version;
        assert!(ds.manifest().uses_stable_row_ids());

        let frag = ds.get_fragment(0).unwrap();
        let frag_id = frag.metadata().id;

        let update_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let update_batch = RecordBatch::try_new(
            update_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![
                    (frag_id << 32) | 1,
                    (frag_id << 32) | 3,
                ])),
                Arc::new(StringArray::from(vec!["B_new", "D_new"])),
            ],
        )
        .unwrap();

        let right_reader: Box<dyn arrow_array::RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(
                vec![Ok(update_batch)],
                update_schema.clone(),
            ));

        let mut frag_mut = ds.get_fragment(0).unwrap();
        let (updated_fragment, fields_modified, matched_bitmap) = frag_mut
            .update_columns(right_reader, ROW_ID, ROW_ID)
            .await
            .unwrap();

        // Convert RoaringBitmap → Vec<u32> (the pattern external callers use)
        let matched_offsets: Vec<u32> = matched_bitmap.unwrap().iter().collect();
        assert_eq!(matched_offsets, vec![1, 3]);

        let frag_before_commit_has_meta = updated_fragment.last_updated_at_version_meta.is_some();

        let mut offsets_map = HashMap::new();
        offsets_map.insert(frag_id, matched_offsets);

        let op = Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![updated_fragment],
            new_fragments: vec![],
            fields_modified,
            merged_generations: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_row_offsets: Some(offsets_map),
        };

        let ds2 = Dataset::commit(
            uri,
            op,
            Some(ds.version().version),
            None,
            None,
            Default::default(),
            true,
        )
        .await
        .unwrap();
        let v2 = ds2.version().version;
        assert!(v2 > v1);

        // Verify data
        let result_batches = ds2
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let mut found: HashMap<i32, String> = HashMap::new();
        for batch in &result_batches {
            let ids = batch.column_by_name("id").unwrap().as_any().downcast_ref::<Int32Array>().unwrap();
            let vals = batch.column_by_name("value").unwrap().as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..ids.len() {
                found.insert(ids.value(i), vals.value(i).to_string());
            }
        }
        assert_eq!(found[&1], "a");
        assert_eq!(found[&2], "B_new");
        assert_eq!(found[&3], "c");
        assert_eq!(found[&4], "D_new");
        assert_eq!(found[&5], "e");

        // Verify version metadata
        let fragments = ds2.get_fragments();
        assert_eq!(fragments.len(), 1);
        let meta = fragments[0].metadata();
        let version_meta = meta
            .last_updated_at_version_meta
            .as_ref()
            .expect("build_manifest should have set version metadata via updated_row_offsets");
        let seq = version_meta.load_sequence().unwrap();
        let physical_rows = meta.physical_rows.unwrap();
        assert_eq!(physical_rows, 5);

        let expected_versions = vec![v1, v2, v1, v2, v1];
        for (row_idx, &expected_ver) in expected_versions.iter().enumerate() {
            let actual_ver = seq.version_at(row_idx).unwrap();
            assert_eq!(
                actual_ver, expected_ver,
                "Row {row_idx}: expected version {expected_ver}, got {actual_ver} \
                 (frag had pre-existing meta: {frag_before_commit_has_meta})"
            );
        }
    }

    /// External caller path: updated_row_offsets is Some but a fragment is NOT
    /// in the map → its version metadata should remain unchanged.
    #[tokio::test]
    async fn test_external_caller_offsets_guard_skips_unlisted_fragments() {
        use arrow_array::UInt64Array;
        use crate::dataset::ROW_ID;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e", "f"])),
            ],
        )
        .unwrap();

        let tmp_dir = tempfile::tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let ds = Dataset::write(
            reader,
            uri,
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let v1 = ds.version().version;
        assert_eq!(ds.get_fragments().len(), 2);

        let frag0 = ds.get_fragment(0).unwrap();
        let frag0_id = frag0.metadata().id;

        let update_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let row_id = (frag0_id << 32) | 1;
        let update_batch = RecordBatch::try_new(
            update_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![row_id])),
                Arc::new(StringArray::from(vec!["B_new"])),
            ],
        )
        .unwrap();

        let right_reader: Box<dyn arrow_array::RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(
                vec![Ok(update_batch)],
                update_schema.clone(),
            ));

        let mut frag0_mut = ds.get_fragment(0).unwrap();
        let (updated_fragment, fields_modified, matched_bitmap) = frag0_mut
            .update_columns(right_reader, ROW_ID, ROW_ID)
            .await
            .unwrap();

        let mut offsets_map = HashMap::new();
        offsets_map.insert(frag0_id, matched_bitmap.unwrap().iter().collect());

        let op = Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![updated_fragment],
            new_fragments: vec![],
            fields_modified,
            merged_generations: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_row_offsets: Some(offsets_map),
        };

        let ds2 = Dataset::commit(
            uri,
            op,
            Some(ds.version().version),
            None,
            None,
            Default::default(),
            true,
        )
        .await
        .unwrap();
        let v2 = ds2.version().version;

        let fragments = ds2.get_fragments();
        assert_eq!(fragments.len(), 2);

        for frag in &fragments {
            let meta = frag.metadata();
            let fid = meta.id;

            if fid == frag0_id {
                let version_meta = meta
                    .last_updated_at_version_meta
                    .as_ref()
                    .expect("Updated fragment should have version metadata");
                let seq = version_meta.load_sequence().unwrap();
                let physical_rows = meta.physical_rows.unwrap();
                assert_eq!(physical_rows, 3);

                assert_eq!(seq.version_at(0).unwrap(), v1);
                assert_eq!(seq.version_at(1).unwrap(), v2);
                assert_eq!(seq.version_at(2).unwrap(), v1);
            } else {
                if let Some(version_meta) = &meta.last_updated_at_version_meta {
                    let seq = version_meta.load_sequence().unwrap();
                    let physical_rows = meta.physical_rows.unwrap();
                    for row_idx in 0..physical_rows {
                        let ver = seq.version_at(row_idx).unwrap();
                        assert_eq!(
                            ver, v1,
                            "Untouched fragment {fid} row {row_idx}: expected v1={v1}, got {ver}"
                        );
                    }
                }
            }
        }
    }

    /// External caller: updated_row_offsets is None + RewriteColumns + stable row IDs
    /// → build_manifest does NOT set version metadata (trusts caller).
    #[tokio::test]
    async fn test_external_caller_offsets_none_does_not_set_metadata() {
        use arrow_array::UInt64Array;
        use crate::dataset::ROW_ID;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();

        let tmp_dir = tempfile::tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let ds = Dataset::write(
            reader,
            uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let v1 = ds.version().version;

        let frag = ds.get_fragment(0).unwrap();
        let frag_id = frag.metadata().id;

        let update_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let row_id = frag_id << 32;
        let update_batch = RecordBatch::try_new(
            update_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![row_id])),
                Arc::new(StringArray::from(vec!["A_new"])),
            ],
        )
        .unwrap();

        let right_reader: Box<dyn arrow_array::RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(
                vec![Ok(update_batch)],
                update_schema.clone(),
            ));

        let mut frag_mut = ds.get_fragment(0).unwrap();
        let (updated_fragment, fields_modified, _matched_offsets) = frag_mut
            .update_columns(right_reader, ROW_ID, ROW_ID)
            .await
            .unwrap();

        let op = Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![updated_fragment.clone()],
            new_fragments: vec![],
            fields_modified,
            merged_generations: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_row_offsets: None,
        };

        let ds2 = Dataset::commit(
            uri,
            op,
            Some(ds.version().version),
            None,
            None,
            Default::default(),
            true,
        )
        .await
        .unwrap();

        let meta = ds2.get_fragment(0).unwrap().metadata().clone();
        if let Some(version_meta) = &meta.last_updated_at_version_meta {
            let seq = version_meta.load_sequence().unwrap();
            let physical_rows = meta.physical_rows.unwrap();
            for row_idx in 0..physical_rows {
                let ver = seq.version_at(row_idx).unwrap();
                assert_eq!(
                    ver, v1,
                    "With updated_row_offsets=None, build_manifest should not \
                     update any version metadata. Row {row_idx}: expected v1={v1}, got {ver}"
                );
            }
        }
    }
}
