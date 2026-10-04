from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    # Type-only: importing OpenWAM pulls in its video/VLM backbones.
    from openwam.model.architectures import BaseWAMArchitecture


class WAMEncoder:
    """Frozen OpenWAM DiT as the observation encoder for the DSRL actor and critics.

    Runs the WAM's video DiT on the clean first frame only, with the text prompt
    and the proprio context token as cross-attention context, and returns the
    mean-pooled first-frame tokens after ``layer`` DiT blocks, followed by the
    explicit (normalized) proprio vector:

        features = [mean_tokens(h_layer(first_frame | text, proprio)), proprio]

    With ``video_attention_mask_mode: first_frame_causal`` the first-frame tokens
    attend only to themselves (never to the noisy future frames, and in every
    ``attention_mask_mode`` never to the action tokens), and a TI2V backbone
    conditions them on t = 0. Their hidden states inside ``generate`` are
    therefore the same at every denoising step and for any noise, and this
    standalone pass reproduces them.

    Inputs are the keyword arguments of ``BaseWAMArchitecture.generate`` for the
    observation (``prompt``, ``first_frame_image``, ``proprio`` in model space,
    ``height``, ``width``, ...); arguments only generation needs are ignored, so
    one dict per observation serves both the encoder and the WAM.
    """

    def __init__(self, wam: "BaseWAMArchitecture", layer: int = 20):
        vb = wam.video_backbone
        if vb is None:
            raise ValueError("WAMEncoder needs a WAM with a video backbone")
        if not 1 <= layer <= vb.num_layers:
            raise ValueError(f"layer must be in [1, {vb.num_layers}], got {layer}")
        if vb.video_attention_mask_mode != "first_frame_causal":
            raise ValueError(
                "WAMEncoder needs video_attention_mask_mode='first_frame_causal'; with "
                f"{vb.video_attention_mask_mode!r} the first frame attends to the noisy future frames inside "
                "generate, so a first-frame-only pass would not match what the WAM computes."
            )
        if not wam.uses_proprioception or wam.proprio_dim <= 0:
            raise ValueError("WAMEncoder needs a WAM trained with proprio (use_proprioception=true)")
        self.wam = wam
        self.layer = layer
        self.feature_dim = vb.dim + wam.proprio_dim

    @torch.no_grad()
    def tokens(
        self,
        *,
        prompt: str,
        first_frame_image,
        proprio,
        height: int = 384,
        width: int = 320,
        seed: int = 42,
        tiled: bool = True,
        prompt_embed_cache: dict | None = None,
        all_layers: bool = False,
        **_generation_only,
    ) -> torch.Tensor | list[torch.Tensor]:
        """First-frame DiT tokens (1, num_tokens, dim) after ``layer`` blocks.

        With ``all_layers`` returns the tokens after each of blocks 1..``layer``.
        ``prompt_embed_cache`` (a dict) caches the text embedding per prompt.
        """
        wam = self.wam
        vb = wam.video_backbone
        device, dtype = wam.device, wam.dtype
        wam.eval()

        # One video frame -> one latent frame, which is the clean observation.
        inputs = vb.preprocess_input_for_inference(
            prompt=prompt,
            first_frame_image=first_frame_image,
            num_frames=1,
            height=height,
            width=width,
            seed=seed,
            tiled=tiled,
            prompt_embed_cache=prompt_embed_cache,
        )
        ref_latents = inputs.get("first_frame_latents")
        if ref_latents is None:
            raise ValueError("WAMEncoder needs a TI2V backbone and a first_frame_image (no clean first-frame latents)")
        # Same clean-frame replacement as generate().
        latents = inputs["latents"].clone()
        latents[:, :, : ref_latents.shape[2]] = ref_latents
        inputs["latents"] = latents

        # Same conditioning as the WAM's forward(): proprio as an extra context token,
        # per-token time modulation with the clean prefix pinned to t = 0.
        proprio = torch.as_tensor(proprio).to(device=device, dtype=dtype).reshape(1, -1)
        inputs = wam._append_proprio_context_token(inputs, proprio)
        inputs.setdefault("force_per_token_t_mod", True)
        inputs.setdefault("zero_clean_prefix_t_mod", True)
        timestep = torch.zeros(1, dtype=dtype, device=device)

        state = vb.prepare(timestep=timestep, **inputs)
        per_layer = []
        for block_id in range(self.layer):
            state = vb.run_block(block_id, state)
            if all_layers:
                per_layer.append(state.hidden_states.clone())
        return per_layer if all_layers else state.hidden_states

    @torch.no_grad()
    def __call__(self, **wam_inputs) -> torch.Tensor:
        """Feature vector (feature_dim,) in float32: pooled first-frame tokens, then proprio."""
        pooled = self.tokens(**wam_inputs).mean(dim=1).reshape(-1)
        proprio = torch.as_tensor(wam_inputs["proprio"]).reshape(-1).to(pooled.device)
        return torch.cat([pooled.float(), proprio.float()])
