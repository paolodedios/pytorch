#pragma once

// Shared between Convolution.metal and operations/Convolution.mm. conv3d_mpp
// bakes K*/S*/D* into template args; conv3d_simd reads them from here.
struct Conv3dDims {
  int C, H, W, O;
  int HO, WO, NB;
  int PADX, PADY;
  int CG, OG, OGT;
  int D, DO, PADZ;
  int KD, KH, KW;
  int SZ, SY, SX;
  int DZ, DY, DX;
  int HAS_BIAS, OUT_NCDHW;
};
