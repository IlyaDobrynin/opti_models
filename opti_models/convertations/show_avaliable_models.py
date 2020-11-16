from opti_models.models import t_models

if __name__ == '__main__':
    torchvision_backbones, opti_backbones = t_models.show_available_backbones()
    print('Torchvision models: {}\n\nOpti_models backbones: {}\n'.format(torchvision_backbones, opti_backbones))
