



source_keys = {
    2: ["chosen", "rejected"],
    3: ["chosen", "med_0", "rejected"],
    4: ["chosen", "med_0", "med_1", "rejected"],
    5: ["chosen", "med_0", "med_1", "med_2", "rejected"],
}

source_labels = {
    2: ["chosen_labels", "rejected_labels"],
    3: ["chosen_labels", "med_0_labels", "rejected_labels"],
    4: ["chosen_labels", "med_0_labels", "med_1_labels", "rejected_labels"],
    5: ["chosen_labels", "med_0_labels", "med_1_labels", "med_2_labels", "rejected_labels"],
}

source_input_idss = {
    2: ["chosen_input_ids", "rejected_input_ids"],
    3: ["chosen_input_ids", "med_0_input_ids", "rejected_input_ids"],
    4: ["chosen_input_ids", "med_0_input_ids", "med_1_input_ids", "rejected_input_ids"],
    5: ["chosen_input_ids", "med_0_input_ids", "med_1_input_ids", "med_2_input_ids", "rejected_input_ids"],
}

source_loss_turbos = {
    2: ["chosen_loss_turbo", "rejected_loss_turbo"],
    3: ["chosen_loss_turbo", "med_0_loss_turbo", "rejected_loss_turbo"],
    4: ["chosen_loss_turbo", "med_0_loss_turbo", "med_1_loss_turbo", "rejected_loss_turbo"],
    5: ["chosen_loss_turbo", "med_0_loss_turbo", "med_1_loss_turbo", "med_2_loss_turbo", "rejected_loss_turbo"],
}

source_attention_masks = {
    2: ["chosen_attention_mask", "rejected_attention_mask"],
    3: ["chosen_attention_mask", "med_0_attention_mask", "rejected_attention_mask"],
    4: ["chosen_attention_mask", "med_0_attention_mask", "med_1_attention_mask", "rejected_attention_mask"],
    5: ["chosen_attention_mask", "med_0_attention_mask", "med_1_attention_mask", "med_2_attention_mask", "rejected_attention_mask"],
}

# source_keys = {
#     2: ["chosen", "rejected"],
#     3: ["chosen", "med_0", "rejected"],
#     4: ["chosen", "med_0", "med_1", "rejected"],
#     5: ["chosen", "med_0", "med_1", "med_2", "rejected"],
# }
# print(source_keys)