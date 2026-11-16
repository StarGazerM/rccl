use crate::{DeviceReadAt, DeviceSliceRead, DeviceInterleavedForRange, ForColumn, ForFrame, for_words, pack_for_word};

#[test]
fn interleaved_for_cpu_oracle() {
    for (block_rows, lanes) in [(256u32, 8u32), (256, 32), (1024, 32)] {
        for bits in 0..=64u32 {
            let mask = if bits == 0 { 0 } else { u64::MAX >> (64 - bits) };
            for base in [0, 1u64.min(u64::MAX - mask), u64::MAX - mask] {
                // Every partial-block row count, not just word-aligned tails.
                for rows in 0..=block_rows {
                    let mut seed = 0x7816_4839_3749_a531u64;
                    let values: Vec<_> = (0..rows).map(|r| {
                        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                        base + if r == 0 { 0 } else if r == 1 { mask } else { seed & mask }
                    }).collect();
                    let words = for_words(rows, lanes, bits) as usize;
                    let input = DeviceSliceRead::new(values.as_ptr());
                    let packed: Vec<_> = (0..words).map(|w| unsafe { pack_for_word(input, rows, lanes, base, bits, w as u32) }).collect();
                    // Independent bit-at-a-time oracle: no extractor or packer
                    // arithmetic is shared with the production implementation.
                    let mut oracle = vec![0u32; words];
                    for (r, &value) in values.iter().enumerate() {
                        let residual = value - base;
                        for b in 0..bits {
                            let bit = if bits <= 32 { r as u64 / lanes as u64 * bits as u64 + b as u64 } else { r as u64 * bits as u64 + b as u64 };
                            let word = if bits <= 32 { bit / 32 * lanes as u64 + r as u64 % lanes as u64 } else { bit / 32 };
                            oracle[word as usize] |= (((residual >> b) & 1) as u32) << (bit % 32);
                        }
                    }
                    assert_eq!(packed, oracle, "pack {block_rows}/{lanes} width {bits}, rows {rows}");
                    let column = ForColumn { payload: packed.as_ptr(), base, bits, rows, lanes };
                    let frames = [ForFrame { base, offset: 0, bits, rows }];
                    let range = DeviceInterleavedForRange { payload: DeviceSliceRead::new(packed.as_ptr()), frames: DeviceSliceRead::new(frames.as_ptr()), block_rows, lanes };
                    let mut sequential = vec![0; rows as usize];
                    for lane in 0..lanes {
                        unsafe { column.visit_lane(lane, 0, rows, 1, |r, v| sequential[r as usize] = v); }
                    }
                    assert_eq!(sequential, values);
                    for r in 0..rows {
                        assert_eq!(unsafe { column.read(r) }, values[r as usize]);
                        assert_eq!(unsafe { range.read_at(r as usize) }, values[r as usize]);
                    }
                    // Partial intervals and stride selection must not decode
                    // an extra row or duplicate another lane's contribution.
                    if rows > 2 {
                        let begin = rows / 3;
                        let end = rows - 1;
                        let mut sum = 0u64;
                        for lane in 0..lanes {
                            unsafe { column.visit_lane(lane, begin, end, 4, |_, v| sum = sum.wrapping_add(v)); }
                        }
                        let expected = (begin..end).step_by(4).fold(0u64, |s, r| s.wrapping_add(values[r as usize]));
                        assert_eq!(sum, expected);
                    }
                }
            }
        }
    }
}

#[test]
fn cooperative_sum_default_preserves_interval() {
    let values: Vec<_> = (0..1040u64).map(|x| x.wrapping_mul(0x9e3779b97f4a7c15)).collect();
    let range = DeviceSliceRead::new(values.as_ptr());
    for (begin, end) in [(0, 0), (0, 1), (1, 31), (7, 1030)] {
        let expected = values[begin..end].iter().fold(0u64, |s, &v| s.wrapping_add(v));
        for lanes in [1, 2, 4, 8, 32, 256] {
            let actual = (0..lanes).fold(0u64, |sum, lane| {
                sum.wrapping_add(unsafe { range.sum_cooperative(begin, end, lane, lanes) })
            });
            assert_eq!(actual, expected);
        }
    }
}
