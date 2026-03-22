from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2 import Qwen2ForCausalLM

from .configuration_dots import DotsVisionConfig, DotsOCRConfig
from .modeling_dots_vision import DotsVisionTransformer


DOTS_VLM_MAX_IMAGES = 200


class DotsOCRForCausalLM(Qwen2ForCausalLM):
    config_class = DotsOCRConfig

    def __init__(self, config: DotsOCRConfig):
        super().__init__(config)

        if isinstance(self.config.vision_config, dict):
            vision_config = DotsVisionConfig(**self.config.vision_config)
            self.config.vision_config = vision_config
        else:
            vision_config = self.config.vision_config

        self.vision_tower = DotsVisionTransformer(vision_config)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_thw: Optional[torch.FloatTensor] = None,
        img_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            assert img_mask is not None
            if grid_thw.shape[0] > DOTS_VLM_MAX_IMAGES:
                print(
                    f"Num image exceeded: {grid_thw.shape[0]} > {DOTS_VLM_MAX_IMAGES}, which may cause FSDP hang"
                )

            vision_embeddings = self.vision_tower(pixel_values, grid_thw)

            true_indices = torch.nonzero(img_mask).squeeze()
            if len(true_indices) > vision_embeddings.size(0):
                print(
                    f"img_mask sum > VE and will be truncated, mask.sum()={len(true_indices)} {vision_embeddings.size(0)=}"
                )
                true_indices = true_indices[: vision_embeddings.size(0)]
                new_img_mask = torch.zeros_like(img_mask, device=img_mask.device)
                new_img_mask[true_indices[:, 0], true_indices[:, 1]] = True
            else:
                new_img_mask = img_mask

            assert (
                vision_embeddings.size(0) == new_img_mask.sum()
            ), f"{vision_embeddings.size(0)=}, {new_img_mask.sum()=}"

            inputs_embeds = inputs_embeds.masked_scatter(
                new_img_mask.to(inputs_embeds.device).unsqueeze(-1).expand_as(inputs_embeds),
                vision_embeddings.to(inputs_embeds.device).type(inputs_embeds.dtype),
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert len(input_ids) >= 1, f"empty input_ids {input_ids.shape=} will cause gradnorm nan"
        if inputs_embeds is None:
            img_mask = input_ids == self.config.image_token_id
            inputs_embeds = self.prepare_inputs_embeds(input_ids, pixel_values, image_grid_thw, img_mask)

        outputs = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache if use_cache is not None else self.config.use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            logits_to_keep=logits_to_keep,
            **loss_kwargs,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values

        return model_inputs
