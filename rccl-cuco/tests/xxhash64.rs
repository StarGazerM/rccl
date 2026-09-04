#[path = "support/xxhash64_vectors.rs"]
mod vectors;

use rccl_cuco::{StableHashBytes, XxHash64};

#[repr(C)]
struct HisaRow<const N: usize> {
    data: [u32; N],
}

// SAFETY: this repr(C) record has one padding-free u32 array field. Its size is
// asserted below and u32 object bytes are little-endian on supported targets.
unsafe impl<const N: usize> StableHashBytes for HisaRow<N> {}

#[test]
fn pinned_cuco_vectors_cover_short_inputs_and_every_block_tail() {
    let aligned = vectors::pattern_bytes();
    let mut unaligned = [0u8; 65];
    unaligned[1..].copy_from_slice(&aligned);

    for length in 0..=63 {
        let seed = vectors::SEEDS[length % vectors::SEEDS.len()];
        let expected = vectors::TAIL_HASHES[length];
        let hash = XxHash64::new(seed);
        assert_eq!(
            hash.hash_bytes(&aligned[..length]),
            expected,
            "aligned length {length} seed {seed:#018x}"
        );
        assert_eq!(
            hash.hash_bytes(&unaligned[1..1 + length]),
            expected,
            "unaligned length {length} seed {seed:#018x}"
        );
    }
}

#[test]
fn pinned_cuco_vectors_cover_primitives_and_hisa_rows() {
    assert_eq!(
        XxHash64::default().hash_value(&0x1234_5678u32),
        vectors::U32_HASH
    );
    assert_eq!(
        XxHash64::new(0x9e37_79b9_7f4a_7c15).hash_value(&0x0123_4567_89ab_cdefu64),
        vectors::U64_HASH
    );

    let row1 = HisaRow { data: [0] };
    let row2 = HisaRow { data: [1, 2] };
    let row4 = HisaRow {
        data: [0, 1, 0x1234_5678, 0xffff_ffff],
    };
    let row7 = HisaRow {
        data: [7, 0, 42, 0x8000_0000, 3, 99, 0xffff_ffff],
    };
    assert_eq!(core::mem::size_of_val(&row1), 4);
    assert_eq!(core::mem::size_of_val(&row2), 8);
    assert_eq!(core::mem::size_of_val(&row4), 16);
    assert_eq!(core::mem::size_of_val(&row7), 28);
    assert_eq!(
        XxHash64::new(0).hash_value(&row1),
        vectors::HISA_ROW_HASHES[0]
    );
    assert_eq!(
        XxHash64::new(1).hash_value(&row2),
        vectors::HISA_ROW_HASHES[1]
    );
    assert_eq!(
        XxHash64::new(0x9e37_79b9_7f4a_7c15).hash_value(&row4),
        vectors::HISA_ROW_HASHES[2]
    );
    assert_eq!(
        XxHash64::new(0).hash_value(&row7),
        vectors::HISA_ROW_HASHES[3]
    );
}

#[test]
fn explicit_encoding_excludes_repr_c_padding() {
    #[repr(C)]
    struct PaddedRecord {
        tag: u8,
        value: u32,
    }

    let record = PaddedRecord {
        tag: 0x5a,
        value: 0x1234_5678,
    };
    assert!(core::mem::size_of::<PaddedRecord>() > 5);

    let mut encoding = [0u8; 5];
    encoding[0] = record.tag;
    encoding[1..].copy_from_slice(&record.value.to_le_bytes());
    assert_eq!(
        XxHash64::new(7).hash_bytes(&encoding),
        XxHash64::new(7).hash_bytes(&[0x5a, 0x78, 0x56, 0x34, 0x12])
    );
}
