"""
Patch Qwen3-VL Conv3d bf16 bug in PyTorch 2.9.
Conv3d with bf16 inputs is ~40,000x slower than fp32 on H200 GPUs.
See: https://github.com/pytorch/pytorch/issues/168167

This patches transformers' Qwen3VLVisionPatchEmbed.forward() to run
the Conv3d in fp32 and cast output back to bf16.

Run this BEFORE importing/loading the model:
    python specforge/patches/patch_conv3d_bf16.py
"""
import site
import os


def patch():
    sp = site.getsitepackages()[0]
    filepath = os.path.join(sp, "transformers", "models", "qwen3_vl", "modeling_qwen3_vl.py")
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} (not found)")
        return

    with open(filepath, "r") as f:
        content = f.read()

    if "Conv3d bf16 is 40000x slower" in content:
        print("Already patched")
        return

    old = """    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states"""

    new = """    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        # Workaround: Conv3d bf16 is 40000x slower than fp32 on PyTorch 2.9 (pytorch#168167)
        hidden_states_f32 = hidden_states.float()
        weight_f32 = self.proj.weight.float()
        bias_f32 = self.proj.bias.float() if self.proj.bias is not None else None
        hidden_states = torch.nn.functional.conv3d(
            hidden_states_f32, weight_f32, bias_f32,
            stride=self.proj.stride, padding=self.proj.padding,
            dilation=self.proj.dilation, groups=self.proj.groups
        ).to(dtype=target_dtype).view(-1, self.embed_dim)
        return hidden_states"""

    if old in content:
        content = content.replace(old, new)
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Patched Conv3d bf16 workaround in {filepath}")
    else:
        print(f"Could not find PatchEmbed.forward pattern in {filepath}")


if __name__ == "__main__":
    patch()
