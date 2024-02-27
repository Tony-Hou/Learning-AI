# [MagicDrive: Street View Generation with Diverse 3D Geometry Control](https://arxiv.org/pdf/2310.02601.pdf)

_February_2024_

tl;dr: employ two strategies (cross-attention and additive encoder branch) to inject text prompt, camera pose, object boxes, and road maps as conditions for generation. We also propose cross-view attention module for multiview consistency.

#### Overall impression

#### Key ideas

#### data normalizer

```
XYZ_MIN = [-200, -300, -20]
XYZ_RANGE = [350, 650, 80]


def normalizer(mode, data):
    if mode == 'cxyz' or mode == 'all-xyz':
        # data in format of (N, 4, 3):
        mins = torch.as_tensor(
            XYZ_MIN, dtype=data.dtype, device=data.device)[None, None]
        divider = torch.as_tensor(
            XYZ_RANGE, dtype=data.dtype, device=data.device)[None, None]
        data = (data - mins) / divider
    elif mode == 'owhr':
        raise NotImplementedError(f"wait for implementation on {mode}")
    else:
        raise NotImplementedError(f"not support {mode}")
    return data
```

#### box embedding
```
self.mode = mode
        if self.mode == 'cxyz':
            input_dims = 3
            output_num = 4  # 4 points
        elif self.mode == 'all-xyz':
            input_dims = 3
            output_num = 8  # 8 points
        elif self.mode == 'owhr':
            raise NotImplementedError("Not sure how to do this.")
        else:
            raise NotImplementedError(f"Wrong mode {mode}")
        self.minmax_normalize = minmax_normalize
        self.use_text_encoder_init = use_text_encoder_init

        self.fourier_embedder = get_embedder(input_dims, embedder_num_freq)
        logging.info(
            f"[ContinuousBBoxWithTextEmbedding] bbox embedder has "
            f"{self.fourier_embedder.out_dim} dims.")

        self.bbox_proj = nn.Linear(
            self.fourier_embedder.out_dim * output_num, proj_dims[0])
        self.second_linear = nn.Sequential(
            nn.Linear(proj_dims[0] + class_token_dim, proj_dims[1]),
            nn.SiLU(),
            nn.Linear(proj_dims[1], proj_dims[2]),
            nn.SiLU(),
            nn.Linear(proj_dims[2], proj_dims[3]),
        )
```