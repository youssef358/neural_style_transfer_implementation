import cv2 as cv
import os
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD = [1, 1, 1]


class ImageProcessor:

    @staticmethod
    def load_image(image_path, size=None):
        if not os.path.exists(image_path):
            raise FileNotFoundError

        image = cv.imread(image_path)[:, :, ::-1]

        if size:
            if isinstance(size, int):
                ratio = size / image.shape[0]
                new_width = int(image.shape[1] * ratio)
                image = cv.resize(image, (new_width, size))
            else:
                image = cv.resize(image, (size[1], size[0]))

        return (image / 255.0).astype(np.float32)

    @staticmethod
    def prepare_image(image_path, size, device):

        image = ImageProcessor.load_image(image_path, size=size)
        transform_queue = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        image = transform_queue(image).to(device).unsqueeze(0)

        return image

    @staticmethod
    def generate_output_image_name(content_image_name, style_image_name, height, content_weight, style_weight,
                                   tv_loss_weight, image_format):
        content_name = os.path.basename(content_image_name).split('.')[0]
        style_name = os.path.basename(style_image_name).split('.')[0]

        prefix = f'{content_name}_{style_name}'

        suffix = f'_h_{str(height)}_cw_{content_weight}_sw_{style_weight}_tv_{tv_loss_weight}{image_format[1]}'

        return prefix + suffix

    @staticmethod
    def save_optimization_result(optimization_image, output_path, config, iteration):
        if iteration == config['num_iterations'] - 1 or (
                config['saving_frequency'] > 0 and iteration % config['saving_frequency'] == 0):
            output_image = optimization_image.squeeze(axis=0).to('cpu').detach().numpy()
            output_image = np.moveaxis(output_image, 0, 2)

            output_image_name = str(iteration).zfill(config['image_format'][0]) + config['image_format'][1] if config[
                                                                                                                   'saving_frequency'] != -1 else ImageProcessor.generate_output_image_name(
                config['content_image_name'], config['style_image_name'], config['height'], config['content_weight'],
                config['style_weight'], config['tv_weight'], config['image_format'])

            final_output_image = np.copy(output_image)
            final_output_image += np.array(IMAGENET_MEAN).reshape((1, 1, 3))
            final_output_image = np.clip(final_output_image, 0, 255).astype('uint8')

            cv.imwrite(os.path.join(output_path, output_image_name), final_output_image[:, :, ::-1])

    @staticmethod
    def show_image(img):
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        plt.imshow(img)
        plt.axis('off')
        plt.show()






