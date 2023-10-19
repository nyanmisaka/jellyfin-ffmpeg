/*
 * Copyright (c) 2020 Yaroslav Pogrebnyak <yyyaroslav@gmail.com>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

template<typename T0, typename T1>
__inline__ __device__ void overlay_func(
    int x_position, int y_position,
    T0* main, int main_linesize,
    int main_adj_x, int main_offset,
    int main_depth, int main_shift,
    T1* overlay, int overlay_linesize,
    int overlay_w, int overlay_h,
    T1* overlay_alpha, int alpha_linesize,
    int alpha_adj_x, int alpha_adj_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= overlay_w + x_position ||
        y >= overlay_h + y_position ||
        x < x_position ||
        y < y_position ) {

        return;
    }

    int overlay_x = x - x_position;
    int overlay_y = y - y_position;

    float alpha = 1.0;
    if (alpha_linesize) {
        alpha = overlay_alpha[alpha_adj_x * overlay_x + alpha_adj_y * overlay_y * alpha_linesize] / 255.0f;
    }

    int main_pos = main_adj_x * x + y * (main_linesize / sizeof(*main)) + (main_adj_x > 1 ? main_offset : 0);
    if (main_depth > 8) {
        T0 overlay_res = (T0)(alpha * overlay[overlay_x + overlay_y * overlay_linesize]) << (main_depth - 8);
        T0 main_res = (T0)((1.0f - alpha) * (main[main_pos] >> main_shift));
        main[main_pos] = (T0)(overlay_res + main_res) << main_shift;
    } else {
        main[main_pos] = alpha * overlay[overlay_x + overlay_y * overlay_linesize] + (1.0f - alpha) * main[main_pos];
    }
}

extern "C" {

#define OVERLAY_VARIANT(NAME, TYPE0) \
__global__ void Overlay_Cuda_ ## NAME( \
    int x_position, int y_position, \
    TYPE0* main, int main_linesize, \
    int main_adj_x, int main_offset, \
    int main_depth, int main_shift, \
    unsigned char* overlay, int overlay_linesize, \
    int overlay_w, int overlay_h, \
    unsigned char* overlay_alpha, int alpha_linesize, \
    int alpha_adj_x, int alpha_adj_y) \
{ \
    overlay_func( \
        x_position, y_position, \
        main, main_linesize, \
        main_adj_x, main_offset, \
        main_depth, main_shift, \
        overlay, overlay_linesize, \
        overlay_w, overlay_h, \
        overlay_alpha, alpha_linesize, \
        alpha_adj_x, alpha_adj_y); \
}

OVERLAY_VARIANT(uchar, unsigned char)
OVERLAY_VARIANT(ushort, unsigned short)

}
