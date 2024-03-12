import os
import configparser


class ChangeDetectionConfig:
    def __init__(self, path='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(path)

    def get_config(self):
        cd_config = {}

        # training setup
        cd_config['output_path'] = self.config['config']['output_path']

        cd_config['no_of_epochs'] = int(self.config['config']['no_of_epochs'])

        cd_config['batch_size'] = int(self.config['config']['batch_size'])
        cd_config['no_of_workers'] = int(self.config['config']['no_of_workers'])

        cd_config['valid_batch'] = int(self.config['config']['valid_batch'])
        cd_config['valid_workers'] = int(self.config['config']['valid_workers'])

        # dataset parameters
        cd_config['dataset_path'] = self.config['config']['dataset_path']
        cd_config['masks_path'] = self.config['config']['masks_path']

        cd_config['train_df_path'] = self.config['config']['train_df_path']
        cd_config['valid_df_path'] = self.config['config']['valid_df_path']
        cd_config['test_df_path'] = self.config['config']['test_df_path']

        cd_config['return_masks'] = self.config['config']['return_masks'] == 'True'

        cd_config['no_of_crops_per_combination'] = int(self.config['config']['no_of_crops_per_combination'])
        cd_config['crop_size'] = int(self.config['config']['crop_size'])

        cd_config['save_model'] = self.config['config']['save_model'] == 'True'
        cd_config['resume_training'] = self.config['config']['resume_training'] == 'True'

        cd_config['model_path'] = self.config['config']['model_path']

        cd_config['tensorboard_path'] = os.path.join(cd_config['output_path'], 'tensorboard')
        return cd_config


if __name__ == '__main__':
    config = ChangeDetectionConfig().get_config()
