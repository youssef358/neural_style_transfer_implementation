import torch

class LossFunctions:

    @staticmethod
    def compute_gram_matrix(img):

        if len(img.size()) == 3:  # If (ch, h, w), add batch dimension
            img = img.unsqueeze(0)

        b, ch, h, w = img.size()
        matrix = img.view(b, ch, h * w)
        matrix_T = matrix.transpose(1, 2)
        gram_matrix = torch.matmul(matrix, matrix_T)
        gram_matrix /= ch * h * w

        return gram_matrix

    @staticmethod
    def compute_total_variation(img):

        b, c, h, w = img.size()
        tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()

        return (tv_h + tv_w) / (c * h * w)

    @staticmethod
    def compute_style_loss(current_representation, target_representation, style_feature_maps):

        style_loss = 0.0
        for layer_name, representation in current_representation.items():

            if layer_name in style_feature_maps:
                style_loss += torch.nn.MSELoss(reduction='sum')(current_representation[layer_name],
                                                                target_representation[layer_name])

        return style_loss / len(style_feature_maps)

    @staticmethod
    def compute_content_loss(current_representation, target_representation, content_feature_map):

        current_content_representation = current_representation[content_feature_map[0]].squeeze(axis=0)

        target_content_representation = target_representation[content_feature_map[0]].squeeze(axis=0)

        content_loss = torch.nn.MSELoss(reduction='mean')(current_content_representation, target_content_representation)

        return content_loss

    @staticmethod
    def compute_total_loss(optimizing_img, current_representation, target_representation, style_feature_maps,
                           content_feature_map, config):

        total_loss = 0.0
        style_loss = LossFunctions.compute_style_loss(current_representation[1], target_representation[1],
                                                      style_feature_maps)

        content_loss = LossFunctions.compute_content_loss(current_representation[0], target_representation[0],
                                                          content_feature_map)

        total_variation_loss = LossFunctions.compute_total_variation(optimizing_img)

        total_loss += config['style_weight'] * style_loss + config['content_weight'] * content_loss + config[
            'tv_weight'] * total_variation_loss

        return total_loss, content_loss, style_loss, total_variation_loss


