# Vision Forward Pass Fix

## Problem Description

The GPT-OSS-Vision model's forward pass was failing when processing multimodal inputs (image + text). The issue was related to position embedding handling when combining vision and text tokens.

## Root Cause

When vision tokens were prepended to text tokens, the `position_ids` were not being updated to account for the new sequence length. This caused:

1. **Position embedding mismatch**: The position embeddings were computed for the original text sequence length, but the actual sequence now included vision tokens
2. **Attention mask issues**: The attention masks weren't properly extended for the combined sequence
3. **RoPE (Rotary Position Embedding) errors**: The rotary embeddings were applied with incorrect position indices

## Solution Applied

### 1. Position IDs Update

In both `modular_gpt_oss_vision.py` and `modeling_gpt_oss_vision.py`, added logic to update `position_ids` when vision tokens are present:

```python
# Update position_ids to account for vision tokens
# Vision tokens should start from position 0, text tokens continue from there
if position_ids is not None:
    # Create new position_ids that start from 0 for vision tokens
    new_position_ids = torch.arange(inputs_embeds.shape[1], device=position_ids.device, dtype=position_ids.dtype)
    new_position_ids = new_position_ids.unsqueeze(0).expand(batch_size, -1)
    position_ids = new_position_ids
else:
    # If position_ids was None, create it for the full sequence
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
```

### 2. VisionAdapter Pooling Fix

Fixed the pooling token handling in the VisionAdapter's fallback conv implementation:

```python
# For fallback conv, there's no class token to replace, so just prepend
vit_embeds = torch.cat([pool, vit_embeds], dim=1)
```

### 3. Attention Mask Extension

Ensured attention masks are properly extended when vision tokens are prepended:

```python
# If attention_mask provided for text tokens only, prepend ones for vision tokens
if attention_mask.dim() == 2 and attention_mask.shape[1] == (inputs_embeds.shape[1] - vis_len):
    vis_ones = torch.ones(batch_size, vis_len, dtype=attention_mask.dtype, device=attention_mask.device)
    attention_mask = torch.cat([vis_ones, attention_mask], dim=1)
```

## Files Modified

1. **`model/gpt_oss_vision/modular_gpt_oss_vision.py`**
   - Fixed position embedding handling in forward method
   - Fixed VisionAdapter pooling logic

2. **`model/gpt_oss_vision/modeling_gpt_oss_vision.py`**
   - Applied same fixes to auto-generated file
   - Ensured consistency between modular and generated versions

3. **`scripts/comprehensive_test.py`**
   - Updated test expectations
   - Added sequence length verification
   - Updated documentation

## Testing

The fix ensures that:

1. **Vision + Text Forward Pass**: Works correctly with proper position embeddings
2. **Sequence Length**: Correctly combines vision and text token counts
3. **Attention Masks**: Properly extended for multimodal sequences
4. **RoPE Embeddings**: Applied with correct position indices

## Expected Behavior

After the fix, the model should:

- Accept both `pixel_values` (images) and `input_ids` (text) as inputs
- Properly concatenate vision and text embeddings
- Apply correct position embeddings to the combined sequence
- Generate outputs with the expected sequence length (vision_tokens + text_tokens)

## Verification

Run the comprehensive test to verify the fix:

```bash
python scripts/comprehensive_test.py
```

The vision forward pass test should now pass successfully.
