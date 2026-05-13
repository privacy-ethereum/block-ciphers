//! Fixsliced implementations of AES-128, AES-192 and AES-256 for WebAssembly
//! 128-bit SIMD (`simd128`).
//!
//! Direct port of the 64-bit fixslice implementation (`crate::soft::fixslice`)
//! to lane-width 128. Every routine here mirrors its `fixslice64` counterpart;
//! only the storage type changes from `u64` (carrying 4 blocks per cell) to
//! `Lane` = `v128` (carrying 8 blocks per cell). The bit-encoding per plane is
//!
//!     position = b + 8·c + 32·r        (b ∈ 0..8, c ∈ 0..4, r ∈ 0..4)
//!
//! so every cell occupies one byte (vs. one nibble in fixslice64) and every
//! within-plane rotation/shift used by the round function and key schedule is
//! a multiple of 8 bits — a single `i8x16.shuffle`.
//!
//! All implementations are fully bitsliced and do not rely on any
//! Look-Up Table (LUT).
//!
//! See the paper at <https://eprint.iacr.org/2020/1123.pdf> for more details
//! on the underlying algorithm.
//!
//! # Author (original C code)
//!
//! Alexandre Adomnicai, Nanyang Technological University, Singapore
//! <alexandre.adomnicai@ntu.edu.sg>
//!
//! Originally licensed MIT. Relicensed as Apache 2.0+MIT with permission.

#![allow(clippy::unreadable_literal, clippy::too_many_arguments)]

use crate::Block;
use cipher::{array::Array, consts::U8};
use core::arch::wasm32::*;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

/// AES block batch size for this implementation
pub(crate) type FixsliceBlocks = U8;

pub(crate) type BatchBlocks = Array<Block, FixsliceBlocks>;

/// AES-128 round keys
pub(crate) type FixsliceKeys128 = [Lane; 88];

/// AES-192 round keys
pub(crate) type FixsliceKeys192 = [Lane; 104];

/// AES-256 round keys
pub(crate) type FixsliceKeys256 = [Lane; 120];

/// 1024-bit internal state (one `Lane` per bit-plane, 8 blocks per cell).
pub(crate) type State = [Lane; 8];

// ============================================================================
// Lane — a v128 carrying one bit-plane of the fixsliced state (8 blocks per
// cell). Mirrors the role of `u64` in fixslice64.
// ============================================================================

#[derive(Copy, Clone)]
pub(crate) struct Lane(v128);

impl Lane {
    #[inline(always)]
    fn zero() -> Self {
        Self(u8x16_splat(0))
    }

    #[inline(always)]
    fn all_ones() -> Self {
        Self(u8x16_splat(0xff))
    }

    #[inline(always)]
    fn from_bytes(b: [u8; 16]) -> Self {
        // SAFETY: `v128` and `[u8; 16]` have identical size and any alignment
        // is acceptable for wasm SIMD loads.
        Self(unsafe { core::mem::transmute::<[u8; 16], v128>(b) })
    }

    #[inline(always)]
    fn to_bytes(self) -> [u8; 16] {
        // SAFETY: see `from_bytes`.
        unsafe { core::mem::transmute::<v128, [u8; 16]>(self.0) }
    }
}

impl Default for Lane {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl BitXor for Lane {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self {
        Self(v128_xor(self.0, r.0))
    }
}
impl BitXorAssign for Lane {
    #[inline(always)]
    fn bitxor_assign(&mut self, r: Self) {
        self.0 = v128_xor(self.0, r.0);
    }
}
impl BitAnd for Lane {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self {
        Self(v128_and(self.0, r.0))
    }
}
impl BitAndAssign for Lane {
    #[inline(always)]
    fn bitand_assign(&mut self, r: Self) {
        self.0 = v128_and(self.0, r.0);
    }
}
impl BitOr for Lane {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self {
        Self(v128_or(self.0, r.0))
    }
}
impl BitOrAssign for Lane {
    #[inline(always)]
    fn bitor_assign(&mut self, r: Self) {
        self.0 = v128_or(self.0, r.0);
    }
}
impl Not for Lane {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(v128_not(self.0))
    }
}

// ============================================================================
// Byte-aligned rotations and shifts on a `Lane` viewed as a single 128-bit
// integer. Mirror the role of `u64::rotate_right`, `<<`, `>>` in fixslice64.
// Within-plane rotation distances scale as `fixslice128_bits = 2 ×
// fixslice64_bits`, and all such distances in the round function and key
// schedule are byte-aligned, so each maps to one `i8x16.shuffle`.
// ============================================================================

macro_rules! ror_b_case {
    ($x:expr, $($i:expr),*) => { Lane(i8x16_shuffle::<$($i),*>($x.0, $x.0)) };
}

/// Rotate the lane as a 128-bit integer right by `N` bytes (= `8·N` bits).
#[inline(always)]
fn ror_b<const N: usize>(x: Lane) -> Lane {
    match N & 15 {
        0 => x,
        1 => ror_b_case!(x, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0),
        2 => ror_b_case!(x, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1),
        3 => ror_b_case!(x, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2),
        4 => ror_b_case!(x, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3),
        5 => ror_b_case!(x, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4),
        6 => ror_b_case!(x, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5),
        7 => ror_b_case!(x, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6),
        8 => ror_b_case!(x, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7),
        9 => ror_b_case!(x, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8),
        10 => ror_b_case!(x, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        11 => ror_b_case!(x, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        12 => ror_b_case!(x, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        13 => ror_b_case!(x, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        14 => ror_b_case!(x, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
        15 => ror_b_case!(x, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
        _ => x,
    }
}

macro_rules! shr_b_case {
    ($x:expr, $z:expr, $($i:expr),*) => { Lane(i8x16_shuffle::<$($i),*>($x.0, $z)) };
}

/// Right-shift the lane as a 128-bit integer by `N` bytes.
#[inline(always)]
fn shr_b<const N: usize>(x: Lane) -> Lane {
    let z = u8x16_splat(0);
    match N {
        0 => x,
        1 => shr_b_case!(x, z, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
        2 => shr_b_case!(x, z, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16),
        3 => shr_b_case!(
            x, z, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16
        ),
        _ => Lane(z),
    }
}

/// Left-shift the lane as a 128-bit integer by `N` bytes.
#[inline(always)]
fn shl_b<const N: usize>(x: Lane) -> Lane {
    let z = u8x16_splat(0);
    match N {
        0 => x,
        1 => shr_b_case!(x, z, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
        2 => shr_b_case!(x, z, 16, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
        3 => shr_b_case!(x, z, 16, 16, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        _ => Lane(z),
    }
}

// ============================================================================
// Mask helpers. fixslice64's nibble-period u64 mask `0xABCDEF...` translates
// to a v128 mask whose 16 bytes are the 16 nibbles of the original constant
// expanded to 0x00/0xff (LSB-nibble first).
// ============================================================================

#[inline(always)]
fn mask(bytes: [u8; 16]) -> Lane {
    Lane::from_bytes(bytes)
}

#[inline(always)]
const fn rep4(p: [u8; 4]) -> [u8; 16] {
    [
        p[0], p[1], p[2], p[3], p[0], p[1], p[2], p[3], p[0], p[1], p[2], p[3], p[0], p[1], p[2],
        p[3],
    ]
}

// ============================================================================
// Fully bitsliced AES-128 key schedule to match the fully-fixsliced representation.
// ============================================================================

pub(crate) fn aes128_key_schedule(key: &[u8; 16]) -> FixsliceKeys128 {
    let mut rkeys = [Lane::zero(); 88];

    bitslice(&mut rkeys[..8], key, key, key, key, key, key, key, key);

    let mut rk_off = 0;
    for rcon in 0..10 {
        memshift32(&mut rkeys, rk_off);
        rk_off += 8;

        sub_bytes(&mut rkeys[rk_off..(rk_off + 8)]);
        sub_bytes_nots(&mut rkeys[rk_off..(rk_off + 8)]);

        if rcon < 8 {
            add_round_constant_bit(&mut rkeys[rk_off..(rk_off + 8)], rcon);
        } else {
            add_round_constant_bit(&mut rkeys[rk_off..(rk_off + 8)], rcon - 8);
            add_round_constant_bit(&mut rkeys[rk_off..(rk_off + 8)], rcon - 7);
            add_round_constant_bit(&mut rkeys[rk_off..(rk_off + 8)], rcon - 5);
            add_round_constant_bit(&mut rkeys[rk_off..(rk_off + 8)], rcon - 4);
        }

        // `ror_distance(1, 3)` = 56 bits = 7 bytes.
        xor_columns::<7>(&mut rkeys, rk_off, 8);
    }

    // Adjust to match fixslicing format (non-compact form).
    for i in (8..72).step_by(32) {
        inv_shift_rows_1(&mut rkeys[i..(i + 8)]);
        inv_shift_rows_2(&mut rkeys[(i + 8)..(i + 16)]);
        inv_shift_rows_3(&mut rkeys[(i + 16)..(i + 24)]);
    }
    inv_shift_rows_1(&mut rkeys[72..80]);

    // Account for NOTs removed from sub_bytes
    for i in 1..11 {
        sub_bytes_nots(&mut rkeys[(i * 8)..(i * 8 + 8)]);
    }

    rkeys
}

// ============================================================================
// Fully bitsliced AES-192 key schedule to match the fully-fixsliced representation.
// ============================================================================

pub(crate) fn aes192_key_schedule(key: &[u8; 24]) -> FixsliceKeys192 {
    let mut rkeys = [Lane::zero(); 104];
    let mut tmp = [Lane::zero(); 8];

    bitslice(
        &mut rkeys[..8],
        &key[..16],
        &key[..16],
        &key[..16],
        &key[..16],
        &key[..16],
        &key[..16],
        &key[..16],
        &key[..16],
    );
    bitslice(
        &mut tmp,
        &key[8..],
        &key[8..],
        &key[8..],
        &key[8..],
        &key[8..],
        &key[8..],
        &key[8..],
        &key[8..],
    );

    // Mask helpers (mirrors of u64 hex constants in fixslice64; nibble period
    // → byte period).
    let m_lo2 = mask(rep4([0xff, 0xff, 0, 0])); // 0x00ff00ff... (col 0..1 of each row)
    let m_hi2 = mask(rep4([0, 0, 0xff, 0xff])); // 0xff00ff00... (col 2..3 of each row)
    let m_col2 = mask(rep4([0, 0, 0xff, 0])); // 0x0f000f00... (col 2 of each row)
    let m_col3 = mask(rep4([0, 0, 0, 0xff])); // 0xf000f000... (col 3 of each row)
    let m_col0 = mask(rep4([0xff, 0, 0, 0])); // 0x000f000f... (col 0 of each row)
    let m_hi3 = mask(rep4([0, 0xff, 0xff, 0xff])); // 0xfff0fff0... (col 1..3 of each row)

    let mut rcon = 0;
    let mut rk_off = 8;

    loop {
        for i in 0..8 {
            rkeys[rk_off + i] =
                (m_lo2 & shr_b::<2>(tmp[i])) | (m_hi2 & shl_b::<2>(rkeys[(rk_off - 8) + i]));
        }

        sub_bytes(&mut tmp);
        sub_bytes_nots(&mut tmp);

        add_round_constant_bit(&mut tmp, rcon);
        rcon += 1;

        for i in 0..8 {
            // `ror_distance(1, 1)` = 40 bits = 5 bytes.
            let mut ti = rkeys[rk_off + i];
            ti ^= m_col2 & ror_b::<5>(tmp[i]);
            ti ^= m_col3 & shl_b::<1>(ti);
            tmp[i] = ti;
        }
        rkeys[rk_off..(rk_off + 8)].copy_from_slice(&tmp);
        rk_off += 8;

        for i in 0..8 {
            let ui = tmp[i];
            let mut ti = (m_lo2 & shr_b::<2>(rkeys[(rk_off - 16) + i])) | (m_hi2 & shl_b::<2>(ui));
            ti ^= m_col0 & shr_b::<3>(ui);
            tmp[i] = ti
                ^ (m_hi3 & shl_b::<1>(ti))
                ^ (m_hi2 & shl_b::<2>(ti))
                ^ (m_col3 & shl_b::<3>(ti));
        }
        rkeys[rk_off..(rk_off + 8)].copy_from_slice(&tmp);
        rk_off += 8;

        sub_bytes(&mut tmp);
        sub_bytes_nots(&mut tmp);

        add_round_constant_bit(&mut tmp, rcon);
        rcon += 1;

        for i in 0..8 {
            // `ror_distance(1, 3)` = 56 bits = 7 bytes.
            let mut ti = (m_lo2 & shr_b::<2>(rkeys[(rk_off - 16) + i]))
                | (m_hi2 & shl_b::<2>(rkeys[(rk_off - 8) + i]));
            ti ^= m_col0 & ror_b::<7>(tmp[i]);
            rkeys[rk_off + i] = ti
                ^ (m_hi3 & shl_b::<1>(ti))
                ^ (m_hi2 & shl_b::<2>(ti))
                ^ (m_col3 & shl_b::<3>(ti));
        }
        rk_off += 8;

        if rcon >= 8 {
            break;
        }

        for i in 0..8 {
            let ui = rkeys[(rk_off - 8) + i];
            let mut ti = rkeys[(rk_off - 16) + i];
            ti ^= m_col2 & shr_b::<1>(ui);
            ti ^= m_col3 & shl_b::<1>(ti);
            tmp[i] = ti;
        }
    }

    // Adjust to match fixslicing format (non-compact form).
    for i in (0..96).step_by(32) {
        inv_shift_rows_1(&mut rkeys[(i + 8)..(i + 16)]);
        inv_shift_rows_2(&mut rkeys[(i + 16)..(i + 24)]);
        inv_shift_rows_3(&mut rkeys[(i + 24)..(i + 32)]);
    }

    // Account for NOTs removed from sub_bytes
    for i in 1..13 {
        sub_bytes_nots(&mut rkeys[(i * 8)..(i * 8 + 8)]);
    }

    rkeys
}

// ============================================================================
// Fully bitsliced AES-256 key schedule to match the fully-fixsliced representation.
// ============================================================================

pub(crate) fn aes256_key_schedule(key: &[u8; 32]) -> FixsliceKeys256 {
    let mut rkeys = [Lane::zero(); 120];

    bitslice(
        &mut rkeys[..8],
        &key[..16],
        &key[..16],
        &key[..16],
        &key[..16],
        &key[..16],
        &key[..16],
        &key[..16],
        &key[..16],
    );
    bitslice(
        &mut rkeys[8..16],
        &key[16..],
        &key[16..],
        &key[16..],
        &key[16..],
        &key[16..],
        &key[16..],
        &key[16..],
        &key[16..],
    );

    let mut rk_off = 8;

    let mut rcon = 0;
    loop {
        memshift32(&mut rkeys, rk_off);
        rk_off += 8;

        sub_bytes(&mut rkeys[rk_off..(rk_off + 8)]);
        sub_bytes_nots(&mut rkeys[rk_off..(rk_off + 8)]);

        add_round_constant_bit(&mut rkeys[rk_off..(rk_off + 8)], rcon);
        // `ror_distance(1, 3)` = 56 bits = 7 bytes.
        xor_columns::<7>(&mut rkeys, rk_off, 16);
        rcon += 1;

        if rcon == 7 {
            break;
        }

        memshift32(&mut rkeys, rk_off);
        rk_off += 8;

        sub_bytes(&mut rkeys[rk_off..(rk_off + 8)]);
        sub_bytes_nots(&mut rkeys[rk_off..(rk_off + 8)]);

        // `ror_distance(0, 3)` = 24 bits = 3 bytes.
        xor_columns::<3>(&mut rkeys, rk_off, 16);
    }

    // Adjust to match fixslicing format (non-compact form).
    for i in (8..104).step_by(32) {
        inv_shift_rows_1(&mut rkeys[i..(i + 8)]);
        inv_shift_rows_2(&mut rkeys[(i + 8)..(i + 16)]);
        inv_shift_rows_3(&mut rkeys[(i + 16)..(i + 24)]);
    }
    inv_shift_rows_1(&mut rkeys[104..112]);

    // Account for NOTs removed from sub_bytes
    for i in 1..15 {
        sub_bytes_nots(&mut rkeys[(i * 8)..(i * 8 + 8)]);
    }

    rkeys
}

// ============================================================================
// Fully-fixsliced AES-128 decryption (the InvShiftRows is completely omitted).
//
// Decrypts eight blocks in-place and in parallel.
// ============================================================================

pub(crate) fn aes128_decrypt(rkeys: &FixsliceKeys128, blocks: &BatchBlocks) -> BatchBlocks {
    let mut state = State::default();

    bitslice(
        &mut state, &blocks[0], &blocks[1], &blocks[2], &blocks[3], &blocks[4], &blocks[5],
        &blocks[6], &blocks[7],
    );

    add_round_key(&mut state, &rkeys[80..]);
    inv_sub_bytes(&mut state);

    inv_shift_rows_2(&mut state);

    let mut rk_off = 72;
    loop {
        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        inv_mix_columns_1(&mut state);
        inv_sub_bytes(&mut state);
        rk_off -= 8;

        if rk_off == 0 {
            break;
        }

        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        inv_mix_columns_0(&mut state);
        inv_sub_bytes(&mut state);
        rk_off -= 8;

        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        inv_mix_columns_3(&mut state);
        inv_sub_bytes(&mut state);
        rk_off -= 8;

        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        inv_mix_columns_2(&mut state);
        inv_sub_bytes(&mut state);
        rk_off -= 8;
    }

    add_round_key(&mut state, &rkeys[..8]);

    inv_bitslice(&state)
}

// ============================================================================
// Fully-fixsliced AES-128 encryption (the ShiftRows is completely omitted).
//
// Encrypts eight blocks in-place and in parallel.
// ============================================================================

pub(crate) fn aes128_encrypt(rkeys: &FixsliceKeys128, blocks: &BatchBlocks) -> BatchBlocks {
    let mut state = State::default();

    bitslice(
        &mut state, &blocks[0], &blocks[1], &blocks[2], &blocks[3], &blocks[4], &blocks[5],
        &blocks[6], &blocks[7],
    );

    add_round_key(&mut state, &rkeys[..8]);

    let mut rk_off = 8;
    loop {
        sub_bytes(&mut state);
        mix_columns_1(&mut state);
        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        rk_off += 8;

        if rk_off == 80 {
            break;
        }

        sub_bytes(&mut state);
        mix_columns_2(&mut state);
        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        rk_off += 8;

        sub_bytes(&mut state);
        mix_columns_3(&mut state);
        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        rk_off += 8;

        sub_bytes(&mut state);
        mix_columns_0(&mut state);
        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        rk_off += 8;
    }

    shift_rows_2(&mut state);

    sub_bytes(&mut state);
    add_round_key(&mut state, &rkeys[80..]);

    inv_bitslice(&state)
}

// ============================================================================
// Fully-fixsliced AES-192 decryption (the InvShiftRows is completely omitted).
//
// Decrypts eight blocks in-place and in parallel.
// ============================================================================

pub(crate) fn aes192_decrypt(rkeys: &FixsliceKeys192, blocks: &BatchBlocks) -> BatchBlocks {
    let mut state = State::default();

    bitslice(
        &mut state, &blocks[0], &blocks[1], &blocks[2], &blocks[3], &blocks[4], &blocks[5],
        &blocks[6], &blocks[7],
    );

    add_round_key(&mut state, &rkeys[96..]);
    inv_sub_bytes(&mut state);

    let mut rk_off = 88;
    loop {
        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        inv_mix_columns_3(&mut state);
        inv_sub_bytes(&mut state);
        rk_off -= 8;

        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        inv_mix_columns_2(&mut state);
        inv_sub_bytes(&mut state);
        rk_off -= 8;

        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        inv_mix_columns_1(&mut state);
        inv_sub_bytes(&mut state);
        rk_off -= 8;

        if rk_off == 0 {
            break;
        }

        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        inv_mix_columns_0(&mut state);
        inv_sub_bytes(&mut state);
        rk_off -= 8;
    }

    add_round_key(&mut state, &rkeys[..8]);

    inv_bitslice(&state)
}

// ============================================================================
// Fully-fixsliced AES-192 encryption (the ShiftRows is completely omitted).
//
// Encrypts eight blocks in-place and in parallel.
// ============================================================================

pub(crate) fn aes192_encrypt(rkeys: &FixsliceKeys192, blocks: &BatchBlocks) -> BatchBlocks {
    let mut state = State::default();

    bitslice(
        &mut state, &blocks[0], &blocks[1], &blocks[2], &blocks[3], &blocks[4], &blocks[5],
        &blocks[6], &blocks[7],
    );

    add_round_key(&mut state, &rkeys[..8]);

    let mut rk_off = 8;
    loop {
        sub_bytes(&mut state);
        mix_columns_1(&mut state);
        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        rk_off += 8;

        sub_bytes(&mut state);
        mix_columns_2(&mut state);
        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        rk_off += 8;

        sub_bytes(&mut state);
        mix_columns_3(&mut state);
        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        rk_off += 8;

        if rk_off == 96 {
            break;
        }

        sub_bytes(&mut state);
        mix_columns_0(&mut state);
        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        rk_off += 8;
    }

    sub_bytes(&mut state);
    add_round_key(&mut state, &rkeys[96..]);

    inv_bitslice(&state)
}

// ============================================================================
// Fully-fixsliced AES-256 decryption (the InvShiftRows is completely omitted).
//
// Decrypts eight blocks in-place and in parallel.
// ============================================================================

pub(crate) fn aes256_decrypt(rkeys: &FixsliceKeys256, blocks: &BatchBlocks) -> BatchBlocks {
    let mut state = State::default();

    bitslice(
        &mut state, &blocks[0], &blocks[1], &blocks[2], &blocks[3], &blocks[4], &blocks[5],
        &blocks[6], &blocks[7],
    );

    add_round_key(&mut state, &rkeys[112..]);
    inv_sub_bytes(&mut state);

    inv_shift_rows_2(&mut state);

    let mut rk_off = 104;
    loop {
        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        inv_mix_columns_1(&mut state);
        inv_sub_bytes(&mut state);
        rk_off -= 8;

        if rk_off == 0 {
            break;
        }

        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        inv_mix_columns_0(&mut state);
        inv_sub_bytes(&mut state);
        rk_off -= 8;

        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        inv_mix_columns_3(&mut state);
        inv_sub_bytes(&mut state);
        rk_off -= 8;

        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        inv_mix_columns_2(&mut state);
        inv_sub_bytes(&mut state);
        rk_off -= 8;
    }

    add_round_key(&mut state, &rkeys[..8]);

    inv_bitslice(&state)
}

// ============================================================================
// Fully-fixsliced AES-256 encryption (the ShiftRows is completely omitted).
//
// Encrypts eight blocks in-place and in parallel.
// ============================================================================

pub(crate) fn aes256_encrypt(rkeys: &FixsliceKeys256, blocks: &BatchBlocks) -> BatchBlocks {
    let mut state = State::default();

    bitslice(
        &mut state, &blocks[0], &blocks[1], &blocks[2], &blocks[3], &blocks[4], &blocks[5],
        &blocks[6], &blocks[7],
    );

    add_round_key(&mut state, &rkeys[..8]);

    let mut rk_off = 8;
    loop {
        sub_bytes(&mut state);
        mix_columns_1(&mut state);
        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        rk_off += 8;

        if rk_off == 112 {
            break;
        }

        sub_bytes(&mut state);
        mix_columns_2(&mut state);
        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        rk_off += 8;

        sub_bytes(&mut state);
        mix_columns_3(&mut state);
        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        rk_off += 8;

        sub_bytes(&mut state);
        mix_columns_0(&mut state);
        add_round_key(&mut state, &rkeys[rk_off..(rk_off + 8)]);
        rk_off += 8;
    }

    shift_rows_2(&mut state);

    sub_bytes(&mut state);
    add_round_key(&mut state, &rkeys[112..]);

    inv_bitslice(&state)
}

// ============================================================================
// Note that the 4 bitwise NOT operations are accounted for here so that this
// is a true inverse of `sub_bytes`.
// ============================================================================

#[inline(always)]
fn inv_sub_bytes(state: &mut [Lane]) {
    debug_assert_eq!(state.len(), 8);

    let u7 = state[0];
    let u6 = state[1];
    let u5 = state[2];
    let u4 = state[3];
    let u3 = state[4];
    let u2 = state[5];
    let u1 = state[6];
    let u0 = state[7];

    let t23 = u0 ^ u3;
    let t8 = u1 ^ t23;
    let m2 = t23 & t8;
    let t4 = u4 ^ t8;
    let t22 = u1 ^ u3;
    let t2 = u0 ^ u1;
    let t1 = u3 ^ u4;
    let t9 = u7 ^ t1;
    let m7 = t22 & t9;
    let t24 = u4 ^ u7;
    let t10 = t2 ^ t24;
    let m14 = t2 & t10;
    let r5 = u6 ^ u7;
    let t3 = t1 ^ r5;
    let t13 = t2 ^ r5;
    let t19 = t22 ^ r5;
    let t17 = u2 ^ t19;
    let t25 = u2 ^ t1;
    let r13 = u1 ^ u6;
    let t20 = t24 ^ r13;
    let m9 = t20 & t17;
    let r17 = u2 ^ u5;
    let t6 = t22 ^ r17;
    let m1 = t13 & t6;
    let y5 = u0 ^ r17;
    let m4 = t19 & y5;
    let m5 = m4 ^ m1;
    let m17 = m5 ^ t24;
    let r18 = u5 ^ u6;
    let t27 = t1 ^ r18;
    let t15 = t10 ^ t27;
    let m11 = t1 & t15;
    let m15 = m14 ^ m11;
    let m21 = m17 ^ m15;
    let m12 = t4 & t27;
    let m13 = m12 ^ m11;
    let t14 = t10 ^ r18;
    let m3 = t14 ^ m1;
    let m16 = m3 ^ m2;
    let m20 = m16 ^ m13;
    let r19 = u2 ^ u4;
    let t16 = r13 ^ r19;
    let t26 = t3 ^ t16;
    let m6 = t3 & t16;
    let m8 = t26 ^ m6;
    let m18 = m8 ^ m7;
    let m22 = m18 ^ m13;
    let m25 = m22 & m20;
    let m26 = m21 ^ m25;
    let m10 = m9 ^ m6;
    let m19 = m10 ^ m15;
    let m23 = m19 ^ t25;
    let m28 = m23 ^ m25;
    let m24 = m22 ^ m23;
    let m30 = m26 & m24;
    let m39 = m23 ^ m30;
    let m48 = m39 & y5;
    let m57 = m39 & t19;
    let m36 = m24 ^ m25;
    let m31 = m20 & m23;
    let m27 = m20 ^ m21;
    let m32 = m27 & m31;
    let m29 = m28 & m27;
    let m37 = m21 ^ m29;
    let m42 = m37 ^ m39;
    let m52 = m42 & t15;
    let m61 = m42 & t1;
    let p0 = m52 ^ m61;
    let p16 = m57 ^ m61;
    let m60 = m37 & t20;
    let m51 = m37 & t17;
    let m33 = m27 ^ m25;
    let m38 = m32 ^ m33;
    let m43 = m37 ^ m38;
    let m49 = m43 & t16;
    let p6 = m49 ^ m60;
    let p13 = m49 ^ m51;
    let m58 = m43 & t3;
    let m50 = m38 & t9;
    let m59 = m38 & t22;
    let p1 = m58 ^ m59;
    let p7 = p0 ^ p1;
    let m34 = m21 & m22;
    let m35 = m24 & m34;
    let m40 = m35 ^ m36;
    let m41 = m38 ^ m40;
    let m45 = m42 ^ m41;
    let m53 = m45 & t27;
    let p8 = m50 ^ m53;
    let p23 = p7 ^ p8;
    let m62 = m45 & t4;
    let p14 = m49 ^ m62;
    let s6 = p14 ^ p23;
    let m54 = m41 & t10;
    let p2 = m54 ^ m62;
    let p22 = p2 ^ p7;
    let s0 = p13 ^ p22;
    let p17 = m58 ^ p2;
    let p15 = m54 ^ m59;
    let m63 = m41 & t2;
    let m44 = m39 ^ m40;
    let m46 = m44 & t6;
    let p5 = m46 ^ m51;
    let p18 = m63 ^ p5;
    let p24 = p5 ^ p7;
    let p12 = m46 ^ m48;
    let s3 = p12 ^ p22;
    let m55 = m44 & t13;
    let p9 = m55 ^ m63;
    let s7 = p9 ^ p16;
    let m47 = m40 & t8;
    let p3 = m47 ^ m50;
    let p19 = p2 ^ p3;
    let s5 = p19 ^ p24;
    let p11 = p0 ^ p3;
    let p26 = p9 ^ p11;
    let m56 = m40 & t23;
    let p4 = m48 ^ m56;
    let p20 = p4 ^ p6;
    let p29 = p15 ^ p20;
    let s1 = p26 ^ p29;
    let p10 = m57 ^ p4;
    let p27 = p10 ^ p18;
    let s4 = p23 ^ p27;
    let p25 = p6 ^ p10;
    let p28 = p11 ^ p25;
    let s2 = p17 ^ p28;

    state[0] = s7;
    state[1] = s6;
    state[2] = s5;
    state[3] = s4;
    state[4] = s3;
    state[5] = s2;
    state[6] = s1;
    state[7] = s0;
}

// ============================================================================
// Bitsliced implementation of the AES Sbox based on Boyar, Peralta and Calik.
//
// See: <http://www.cs.yale.edu/homes/peralta/CircuitStuff/SLP_AES_113.txt>
//
// Note that the 4 bitwise NOTs are moved to the key schedule.
// ============================================================================

fn sub_bytes(state: &mut [Lane]) {
    debug_assert_eq!(state.len(), 8);

    let u7 = state[0];
    let u6 = state[1];
    let u5 = state[2];
    let u4 = state[3];
    let u3 = state[4];
    let u2 = state[5];
    let u1 = state[6];
    let u0 = state[7];

    let y14 = u3 ^ u5;
    let y13 = u0 ^ u6;
    let y12 = y13 ^ y14;
    let t1 = u4 ^ y12;
    let y15 = t1 ^ u5;
    let t2 = y12 & y15;
    let y6 = y15 ^ u7;
    let y20 = t1 ^ u1;
    let y9 = u0 ^ u3;
    let y11 = y20 ^ y9;
    let t12 = y9 & y11;
    let y7 = u7 ^ y11;
    let y8 = u0 ^ u5;
    let t0 = u1 ^ u2;
    let y10 = y15 ^ t0;
    let y17 = y10 ^ y11;
    let t13 = y14 & y17;
    let t14 = t13 ^ t12;
    let y19 = y10 ^ y8;
    let t15 = y8 & y10;
    let t16 = t15 ^ t12;
    let y16 = t0 ^ y11;
    let y21 = y13 ^ y16;
    let t7 = y13 & y16;
    let y18 = u0 ^ y16;
    let y1 = t0 ^ u7;
    let y4 = y1 ^ u3;
    let t5 = y4 & u7;
    let t6 = t5 ^ t2;
    let t18 = t6 ^ t16;
    let t22 = t18 ^ y19;
    let y2 = y1 ^ u0;
    let t10 = y2 & y7;
    let t11 = t10 ^ t7;
    let t20 = t11 ^ t16;
    let t24 = t20 ^ y18;
    let y5 = y1 ^ u6;
    let t8 = y5 & y1;
    let t9 = t8 ^ t7;
    let t19 = t9 ^ t14;
    let t23 = t19 ^ y21;
    let y3 = y5 ^ y8;
    let t3 = y3 & y6;
    let t4 = t3 ^ t2;
    let t17 = t4 ^ y20;
    let t21 = t17 ^ t14;
    let t26 = t21 & t23;
    let t27 = t24 ^ t26;
    let t31 = t22 ^ t26;
    let t25 = t21 ^ t22;
    let t28 = t25 & t27;
    let t29 = t28 ^ t22;
    let z14 = t29 & y2;
    let z5 = t29 & y7;
    let t30 = t23 ^ t24;
    let t32 = t31 & t30;
    let t33 = t32 ^ t24;
    let t35 = t27 ^ t33;
    let t36 = t24 & t35;
    let t38 = t27 ^ t36;
    let t39 = t29 & t38;
    let t40 = t25 ^ t39;
    let t43 = t29 ^ t40;
    let z3 = t43 & y16;
    let tc12 = z3 ^ z5;
    let z12 = t43 & y13;
    let z13 = t40 & y5;
    let z4 = t40 & y1;
    let tc6 = z3 ^ z4;
    let t34 = t23 ^ t33;
    let t37 = t36 ^ t34;
    let t41 = t40 ^ t37;
    let z8 = t41 & y10;
    let z17 = t41 & y8;
    let t44 = t33 ^ t37;
    let z0 = t44 & y15;
    let z9 = t44 & y12;
    let z10 = t37 & y3;
    let z1 = t37 & y6;
    let tc5 = z1 ^ z0;
    let tc11 = tc6 ^ tc5;
    let z11 = t33 & y4;
    let t42 = t29 ^ t33;
    let t45 = t42 ^ t41;
    let z7 = t45 & y17;
    let tc8 = z7 ^ tc6;
    let z16 = t45 & y14;
    let z6 = t42 & y11;
    let tc16 = z6 ^ tc8;
    let z15 = t42 & y9;
    let tc20 = z15 ^ tc16;
    let tc1 = z15 ^ z16;
    let tc2 = z10 ^ tc1;
    let tc21 = tc2 ^ z11;
    let tc3 = z9 ^ tc2;
    let s0 = tc3 ^ tc16;
    let s3 = tc3 ^ tc11;
    let s1 = s3 ^ tc16;
    let tc13 = z13 ^ tc1;
    let z2 = t33 & u7;
    let tc4 = z0 ^ z2;
    let tc7 = z12 ^ tc4;
    let tc9 = z8 ^ tc7;
    let tc10 = tc8 ^ tc9;
    let tc17 = z14 ^ tc10;
    let s5 = tc21 ^ tc17;
    let tc26 = tc17 ^ tc20;
    let s2 = tc26 ^ z17;
    let tc14 = tc4 ^ tc12;
    let tc18 = tc13 ^ tc14;
    let s6 = tc10 ^ tc18;
    let s7 = z12 ^ tc18;
    let s4 = tc14 ^ s3;

    state[0] = s7;
    state[1] = s6;
    state[2] = s5;
    state[3] = s4;
    state[4] = s3;
    state[5] = s2;
    state[6] = s1;
    state[7] = s0;
}

/// NOT operations that are omitted in S-box.
#[inline]
fn sub_bytes_nots(state: &mut [Lane]) {
    debug_assert_eq!(state.len(), 8);
    let ones = Lane::all_ones();
    state[0] ^= ones;
    state[1] ^= ones;
    state[5] ^= ones;
    state[6] ^= ones;
}

// ============================================================================
// Computation of the MixColumns transformation in the fixsliced representation,
// with different rotations used according to the round number mod 4.
//
// Based on Käsper-Schwabe, similar to https://github.com/Ko-/aes-armcortexm.
// ============================================================================

macro_rules! define_mix_columns {
    (
        $name:ident,
        $name_inv:ident,
        $first_rotate:path,
        $second_rotate:path
    ) => {
        #[rustfmt::skip]
        fn $name(state: &mut State) {
            let (a0, a1, a2, a3, a4, a5, a6, a7) = (
                state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]
            );
            let (b0, b1, b2, b3, b4, b5, b6, b7) = (
                $first_rotate(a0),
                $first_rotate(a1),
                $first_rotate(a2),
                $first_rotate(a3),
                $first_rotate(a4),
                $first_rotate(a5),
                $first_rotate(a6),
                $first_rotate(a7),
            );
            let (c0, c1, c2, c3, c4, c5, c6, c7) = (
                a0 ^ b0,
                a1 ^ b1,
                a2 ^ b2,
                a3 ^ b3,
                a4 ^ b4,
                a5 ^ b5,
                a6 ^ b6,
                a7 ^ b7,
            );
            state[0] = b0      ^ c7 ^ $second_rotate(c0);
            state[1] = b1 ^ c0 ^ c7 ^ $second_rotate(c1);
            state[2] = b2 ^ c1      ^ $second_rotate(c2);
            state[3] = b3 ^ c2 ^ c7 ^ $second_rotate(c3);
            state[4] = b4 ^ c3 ^ c7 ^ $second_rotate(c4);
            state[5] = b5 ^ c4      ^ $second_rotate(c5);
            state[6] = b6 ^ c5      ^ $second_rotate(c6);
            state[7] = b7 ^ c6      ^ $second_rotate(c7);
        }

        #[rustfmt::skip]
        fn $name_inv(state: &mut State) {
            let (a0, a1, a2, a3, a4, a5, a6, a7) = (
                state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]
            );
            let (b0, b1, b2, b3, b4, b5, b6, b7) = (
                $first_rotate(a0),
                $first_rotate(a1),
                $first_rotate(a2),
                $first_rotate(a3),
                $first_rotate(a4),
                $first_rotate(a5),
                $first_rotate(a6),
                $first_rotate(a7),
            );
            let (c0, c1, c2, c3, c4, c5, c6, c7) = (
                a0 ^ b0,
                a1 ^ b1,
                a2 ^ b2,
                a3 ^ b3,
                a4 ^ b4,
                a5 ^ b5,
                a6 ^ b6,
                a7 ^ b7,
            );
            let (d0, d1, d2, d3, d4, d5, d6, d7) = (
                a0      ^ c7,
                a1 ^ c0 ^ c7,
                a2 ^ c1,
                a3 ^ c2 ^ c7,
                a4 ^ c3 ^ c7,
                a5 ^ c4,
                a6 ^ c5,
                a7 ^ c6,
            );
            let (e0, e1, e2, e3, e4, e5, e6, e7) = (
                c0      ^ d6,
                c1      ^ d6 ^ d7,
                c2 ^ d0      ^ d7,
                c3 ^ d1 ^ d6,
                c4 ^ d2 ^ d6 ^ d7,
                c5 ^ d3      ^ d7,
                c6 ^ d4,
                c7 ^ d5,
            );
            state[0] = d0 ^ e0 ^ $second_rotate(e0);
            state[1] = d1 ^ e1 ^ $second_rotate(e1);
            state[2] = d2 ^ e2 ^ $second_rotate(e2);
            state[3] = d3 ^ e3 ^ $second_rotate(e3);
            state[4] = d4 ^ e4 ^ $second_rotate(e4);
            state[5] = d5 ^ e5 ^ $second_rotate(e5);
            state[6] = d6 ^ e6 ^ $second_rotate(e6);
            state[7] = d7 ^ e7 ^ $second_rotate(e7);
        }
    }
}

define_mix_columns!(
    mix_columns_0,
    inv_mix_columns_0,
    rotate_rows_1,
    rotate_rows_2
);

define_mix_columns!(
    mix_columns_1,
    inv_mix_columns_1,
    rotate_rows_and_columns_1_1,
    rotate_rows_and_columns_2_2
);

define_mix_columns!(
    mix_columns_2,
    inv_mix_columns_2,
    rotate_rows_and_columns_1_2,
    rotate_rows_2
);

define_mix_columns!(
    mix_columns_3,
    inv_mix_columns_3,
    rotate_rows_and_columns_1_3,
    rotate_rows_and_columns_2_2
);

// ============================================================================
// Byte-aligned `delta_swap` (mirrors fixslice64 `delta_swap_1` whose shifts
// were 4 and 8 bits — both byte-aligned at lane-width 128).
// ============================================================================

#[inline]
fn delta_swap_1<const BY: usize>(a: &mut Lane, mask: Lane) {
    let t = (*a ^ shr_b::<BY>(*a)) & mask;
    *a ^= t ^ shl_b::<BY>(t);
}

/// Applies ShiftRows once on an AES state (or key).
#[inline]
fn shift_rows_1(state: &mut [Lane]) {
    debug_assert_eq!(state.len(), 8);
    let m1 = mask([0, 0, 0, 0, 0xff, 0, 0, 0, 0xff, 0xff, 0, 0, 0, 0xff, 0, 0]);
    let m2 = mask([0, 0, 0, 0, 0xff, 0, 0xff, 0, 0, 0, 0, 0, 0xff, 0, 0xff, 0]);
    for x in state.iter_mut() {
        delta_swap_1::<2>(x, m1);
        delta_swap_1::<1>(x, m2);
    }
}

/// Applies ShiftRows twice on an AES state (or key).
#[inline]
fn shift_rows_2(state: &mut [Lane]) {
    debug_assert_eq!(state.len(), 8);
    let m = mask([0, 0, 0, 0, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0, 0]);
    for x in state.iter_mut() {
        delta_swap_1::<2>(x, m);
    }
}

/// Applies ShiftRows three times on an AES state (or key).
#[inline]
fn shift_rows_3(state: &mut [Lane]) {
    debug_assert_eq!(state.len(), 8);
    let m1 = mask([0, 0, 0, 0, 0, 0xff, 0, 0, 0xff, 0xff, 0, 0, 0xff, 0, 0, 0]);
    let m2 = mask([0, 0, 0, 0, 0xff, 0, 0xff, 0, 0, 0, 0, 0, 0xff, 0, 0xff, 0]);
    for x in state.iter_mut() {
        delta_swap_1::<2>(x, m1);
        delta_swap_1::<1>(x, m2);
    }
}

#[inline(always)]
fn inv_shift_rows_1(state: &mut [Lane]) {
    shift_rows_3(state);
}

#[inline(always)]
fn inv_shift_rows_2(state: &mut [Lane]) {
    shift_rows_2(state);
}

#[inline(always)]
fn inv_shift_rows_3(state: &mut [Lane]) {
    shift_rows_1(state);
}

// ============================================================================
// XOR the columns after the S-box during the key schedule round function.
//
// `idx_xor` refers to the index of the previous round key that is involved in
// the XOR computation (8 and 16 for AES-128 and AES-256, respectively).
// `ROR` is the byte-aligned rotation distance, which varies between the
// different key schedules (= `ror_distance(r, c) / 4` in fixslice64 bits).
// ============================================================================

fn xor_columns<const ROR: usize>(rkeys: &mut [Lane], offset: usize, idx_xor: usize) {
    let m_lo1 = mask(rep4([0xff, 0, 0, 0])); // 0x000f000f... (cell at row 0)
    let m_hi3 = mask(rep4([0, 0xff, 0xff, 0xff])); // 0xfff0fff0... (cells row 1..3)
    let m_hi2 = mask(rep4([0, 0, 0xff, 0xff])); // 0xff00ff00... (cells row 2..3)
    let m_hi1 = mask(rep4([0, 0, 0, 0xff])); // 0xf000f000... (cell at row 3)
    for i in 0..8 {
        let off_i = offset + i;
        let rk = rkeys[off_i - idx_xor] ^ (m_lo1 & ror_b::<ROR>(rkeys[off_i]));
        rkeys[off_i] =
            rk ^ (m_hi3 & shl_b::<1>(rk)) ^ (m_hi2 & shl_b::<2>(rk)) ^ (m_hi1 & shl_b::<3>(rk));
    }
}

// ============================================================================
// Bitslice eight 128-bit input blocks into a 1024-bit internal state.
//
// Mirrors fixslice64's `bitslice` but for 8 blocks per state instead of 4
// (since each `Lane` carries 8 blocks per cell vs. 4 in u64). Because each
// block fully occupies one lane's worth of bits, the read-and-reorder step
// from fixslice64 isn't needed: we just load each block directly, do a 4×4
// byte transpose to put the AES state in row-major form within the lane, and
// then perform the 8×8 bit transpose that swaps "block" and "bit-position"
// across the 8 lanes.
// ============================================================================

fn bitslice(
    output: &mut [Lane],
    in0: &[u8],
    in1: &[u8],
    in2: &[u8],
    in3: &[u8],
    in4: &[u8],
    in5: &[u8],
    in6: &[u8],
    in7: &[u8],
) {
    debug_assert_eq!(output.len(), 8);
    debug_assert_eq!(in0.len(), 16);
    debug_assert_eq!(in1.len(), 16);
    debug_assert_eq!(in2.len(), 16);
    debug_assert_eq!(in3.len(), 16);
    debug_assert_eq!(in4.len(), 16);
    debug_assert_eq!(in5.len(), 16);
    debug_assert_eq!(in6.len(), 16);
    debug_assert_eq!(in7.len(), 16);

    fn load_block(input: &[u8]) -> Lane {
        let mut b = [0u8; 16];
        b.copy_from_slice(input);
        Lane::from_bytes(b)
    }

    // Load one block per lane and transpose its 4×4 byte matrix so that the
    // within-lane index is `c b2 b1 b0 r1 r0 p2 p1 p0` (column on top).
    // After the bit transpose below, plane-index `p2 p1 p0` floats to the top
    // (selecting which output lane) and block-index `b2 b1 b0` lives in the
    // low 3 bits of every byte — the desired fixslice128 encoding.
    let mut t = [
        transpose_4x4_bytes(load_block(in0)),
        transpose_4x4_bytes(load_block(in1)),
        transpose_4x4_bytes(load_block(in2)),
        transpose_4x4_bytes(load_block(in3)),
        transpose_4x4_bytes(load_block(in4)),
        transpose_4x4_bytes(load_block(in5)),
        transpose_4x4_bytes(load_block(in6)),
        transpose_4x4_bytes(load_block(in7)),
    ];
    transpose_8x8_bits(&mut t);
    output.copy_from_slice(&t);
}

// ============================================================================
// Un-bitslice a 1024-bit internal state into eight 128-bit blocks of output.
// ============================================================================

#[inline(always)]
fn inv_bitslice(input: &[Lane]) -> BatchBlocks {
    debug_assert_eq!(input.len(), 8);

    let mut t = [
        input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7],
    ];
    transpose_8x8_bits(&mut t);
    let mut output = BatchBlocks::default();
    for k in 0..8 {
        output[k] = transpose_4x4_bytes(t[k]).to_bytes().into();
    }
    output
}

/// 4×4 byte transpose within one 16-byte lane (`out[c + 4·r] = in[4·c + r]`).
/// Self-inverse.
#[inline(always)]
fn transpose_4x4_bytes(x: Lane) -> Lane {
    Lane(i8x16_shuffle::<
        0,
        4,
        8,
        12,
        1,
        5,
        9,
        13,
        2,
        6,
        10,
        14,
        3,
        7,
        11,
        15,
    >(x.0, x.0))
}

/// Per-byte sub-byte swap: at each byte position, the bits selected by `mask`
/// in `a` are exchanged with the bits at `shift` positions higher in `b`.
/// Mirrors fixslice64's `delta_swap_2` but with byte-wise sub-byte shifts
/// instead of whole-register bit shifts.
#[inline(always)]
fn bit_swap_pair<const SHIFT: u32>(a: &mut Lane, b: &mut Lane, msk: Lane) {
    let t = (*a ^ Lane(u8x16_shr(b.0, SHIFT))) & msk;
    *a ^= t;
    *b ^= Lane(u8x16_shl(t.0, SHIFT));
}

/// 8×8 bit transpose across 8 lanes (independent at each of the 16 byte
/// positions). Self-inverse.
#[inline]
fn transpose_8x8_bits(t: &mut [Lane; 8]) {
    let [t0, t1, t2, t3, t4, t5, t6, t7] = t;
    let m1 = mask([0x55; 16]);
    bit_swap_pair::<1>(t1, t0, m1);
    bit_swap_pair::<1>(t3, t2, m1);
    bit_swap_pair::<1>(t5, t4, m1);
    bit_swap_pair::<1>(t7, t6, m1);
    let m2 = mask([0x33; 16]);
    bit_swap_pair::<2>(t2, t0, m2);
    bit_swap_pair::<2>(t3, t1, m2);
    bit_swap_pair::<2>(t6, t4, m2);
    bit_swap_pair::<2>(t7, t5, m2);
    let m4 = mask([0x0f; 16]);
    bit_swap_pair::<4>(t4, t0, m4);
    bit_swap_pair::<4>(t5, t1, m4);
    bit_swap_pair::<4>(t6, t2, m4);
    bit_swap_pair::<4>(t7, t3, m4);
}

/// Copy 8 lanes (= one round key) within `buffer` from `src_offset` to
/// `src_offset + 8`.
fn memshift32(buffer: &mut [Lane], src_offset: usize) {
    debug_assert_eq!(src_offset % 8, 0);
    let dst_offset = src_offset + 8;
    debug_assert!(dst_offset + 8 <= buffer.len());
    for i in (0..8).rev() {
        buffer[dst_offset + i] = buffer[src_offset + i];
    }
}

/// XOR the round key to the internal state. Round keys are expected to be
/// pre-computed and packed in the fixsliced representation.
#[inline]
fn add_round_key(state: &mut State, rkey: &[Lane]) {
    debug_assert_eq!(rkey.len(), 8);
    for (a, b) in state.iter_mut().zip(rkey) {
        *a ^= *b;
    }
}

/// XOR a 1-bit round constant into byte 7 (row 1, column 3) of bit-plane
/// `bit` — the lane-128 analogue of `state[bit] ^= 0xf0000000` in fixslice64.
#[inline(always)]
fn add_round_constant_bit(state: &mut [Lane], bit: usize) {
    let m = mask([0, 0, 0, 0, 0, 0, 0, 0xff, 0, 0, 0, 0, 0, 0, 0, 0]);
    state[bit] ^= m;
}

#[inline(always)]
fn rotate_rows_1(x: Lane) -> Lane {
    // ror_distance(1, 0) = 32 bits = 4 bytes
    ror_b::<4>(x)
}

#[inline(always)]
fn rotate_rows_2(x: Lane) -> Lane {
    // ror_distance(2, 0) = 64 bits = 8 bytes
    ror_b::<8>(x)
}

// The compound `(ror_A & MA) | (ror_B & MB)` patterns below have disjoint
// byte masks and so collapse to a single `i8x16.shuffle` selecting per-byte
// sources.

#[inline(always)]
fn rotate_rows_and_columns_1_1(x: Lane) -> Lane {
    // (ror::<5> & 0x0fff0fff…) | (ror::<1> & 0xf000f000…)
    Lane(i8x16_shuffle::<
        5,
        6,
        7,
        4,
        9,
        10,
        11,
        8,
        13,
        14,
        15,
        12,
        1,
        2,
        3,
        0,
    >(x.0, x.0))
}

#[inline(always)]
fn rotate_rows_and_columns_1_2(x: Lane) -> Lane {
    // (ror::<6> & 0x00ff00ff…) | (ror::<2> & 0xff00ff00…)
    Lane(i8x16_shuffle::<
        6,
        7,
        4,
        5,
        10,
        11,
        8,
        9,
        14,
        15,
        12,
        13,
        2,
        3,
        0,
        1,
    >(x.0, x.0))
}

#[inline(always)]
fn rotate_rows_and_columns_1_3(x: Lane) -> Lane {
    // (ror::<7> & 0x000f000f…) | (ror::<3> & 0xfff0fff0…)
    Lane(i8x16_shuffle::<
        7,
        4,
        5,
        6,
        11,
        8,
        9,
        10,
        15,
        12,
        13,
        14,
        3,
        0,
        1,
        2,
    >(x.0, x.0))
}

#[inline(always)]
fn rotate_rows_and_columns_2_2(x: Lane) -> Lane {
    // (ror::<10> & 0x00ff00ff…) | (ror::<6> & 0xff00ff00…)
    Lane(i8x16_shuffle::<
        10,
        11,
        8,
        9,
        14,
        15,
        12,
        13,
        2,
        3,
        0,
        1,
        6,
        7,
        4,
        5,
    >(x.0, x.0))
}

// ============================================================================
// Low-level "hazmat" AES functions.
//
// Mirrors `crate::soft::fixslice::hazmat`. Since each `State` carries 8 blocks
// natively, the `_par` variants run in a single bitslice/round/inv_bitslice
// pass with no per-chunk iteration.
// ============================================================================
#[cfg(feature = "hazmat")]
pub(crate) mod hazmat {
    use super::{
        State, bitslice, inv_bitslice, inv_mix_columns_0, inv_shift_rows_1, inv_sub_bytes,
        mix_columns_0, shift_rows_1, sub_bytes, sub_bytes_nots,
    };
    use crate::hazmat::{Block, Block8};

    fn xor_in_place(dst: &mut Block, src: &Block) {
        for (a, b) in dst.iter_mut().zip(src.as_slice()) {
            *a ^= *b;
        }
    }

    /// Bitslice a single block by broadcasting it to all 8 block-positions.
    fn bitslice_block(block: &Block) -> State {
        let mut state = State::default();
        bitslice(
            &mut state, block, block, block, block, block, block, block, block,
        );
        state
    }

    /// Un-bitslice a single block (takes the first of the 8 identical outputs).
    fn inv_bitslice_block(block: &mut Block, state: &State) {
        block.copy_from_slice(&inv_bitslice(state)[0]);
    }

    /// AES cipher (encrypt) round function.
    #[inline]
    pub(crate) fn cipher_round(block: &mut Block, round_key: &Block) {
        let mut state = bitslice_block(block);
        sub_bytes(&mut state);
        sub_bytes_nots(&mut state);
        shift_rows_1(&mut state);
        mix_columns_0(&mut state);
        inv_bitslice_block(block, &state);
        xor_in_place(block, round_key);
    }

    /// AES cipher (encrypt) round function: parallel version.
    #[inline]
    pub(crate) fn cipher_round_par(blocks: &mut Block8, round_keys: &Block8) {
        let mut state = State::default();
        bitslice(
            &mut state, &blocks[0], &blocks[1], &blocks[2], &blocks[3], &blocks[4], &blocks[5],
            &blocks[6], &blocks[7],
        );
        sub_bytes(&mut state);
        sub_bytes_nots(&mut state);
        shift_rows_1(&mut state);
        mix_columns_0(&mut state);
        let res = inv_bitslice(&state);
        for i in 0..8 {
            blocks[i] = res[i];
            xor_in_place(&mut blocks[i], &round_keys[i]);
        }
    }

    /// AES equivalent inverse cipher (decrypt) round function.
    #[inline]
    pub(crate) fn equiv_inv_cipher_round(block: &mut Block, round_key: &Block) {
        let mut state = bitslice_block(block);
        sub_bytes_nots(&mut state);
        inv_sub_bytes(&mut state);
        inv_shift_rows_1(&mut state);
        inv_mix_columns_0(&mut state);
        inv_bitslice_block(block, &state);
        xor_in_place(block, round_key);
    }

    /// AES equivalent inverse cipher (decrypt) round function: parallel version.
    #[inline]
    pub(crate) fn equiv_inv_cipher_round_par(blocks: &mut Block8, round_keys: &Block8) {
        let mut state = State::default();
        bitslice(
            &mut state, &blocks[0], &blocks[1], &blocks[2], &blocks[3], &blocks[4], &blocks[5],
            &blocks[6], &blocks[7],
        );
        sub_bytes_nots(&mut state);
        inv_sub_bytes(&mut state);
        inv_shift_rows_1(&mut state);
        inv_mix_columns_0(&mut state);
        let res = inv_bitslice(&state);
        for i in 0..8 {
            blocks[i] = res[i];
            xor_in_place(&mut blocks[i], &round_keys[i]);
        }
    }

    /// AES mix columns function.
    #[inline]
    pub(crate) fn mix_columns(block: &mut Block) {
        let mut state = bitslice_block(block);
        mix_columns_0(&mut state);
        inv_bitslice_block(block, &state);
    }

    /// AES inverse mix columns function.
    #[inline]
    pub(crate) fn inv_mix_columns(block: &mut Block) {
        let mut state = bitslice_block(block);
        inv_mix_columns_0(&mut state);
        inv_bitslice_block(block, &state);
    }
}
