import torch
from typing import Optional, Union, Tuple, Any
from transformers.modeling_outputs import ModelOutput
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, PreTrainedModel
from datasets import load_from_disk
from dataclasses import dataclass

@dataclass
class GPT2ForGenerationAndClassificationOutput(ModelOutput):
    loss: Any =  None
    logits: Any =  None    
    hidden_states: Any = None
    attentions: Any =  None
    next_token_logits: Any =  None

# todo don't forget to add the padding token to the tokenizer but to the left instead of to the right.

class GPT2ForClassificationAndGeneration(PreTrainedModel):
    def __init__(self, model_name, num_classes,config=None):
        super(GPT2ForClassificationAndGeneration, self).__init__(config)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)  # For text generation
        self.classification_head = nn.Linear(self.gpt2.config.n_embd, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_classes

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], GPT2ForGenerationAndClassificationOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt2(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        #Todo check this- what should it be
        # Todo check maybe it needs to be outputs["hidden_states"]
        #todo make that this is indeed the hidden state of the LAST layer
        sequence_output = outputs.hidden_states[-1]

        sequence_output = self.dropout(sequence_output)
        logits = self.classification_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return GPT2ForGenerationAndClassificationOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            next_token_logits=outputs.logits
        )
   