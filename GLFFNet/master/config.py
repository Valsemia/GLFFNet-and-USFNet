# coding: utf-8
sbu_training_root = r'Datasets/ISTD_Dataset/train'
sbu_testing_root = r'Datasets/ISTD_Dataset/test'
sbu_validation_root = r''

resnext_101_32_path = r'resnext/resnext_101_32x4d.pth'
pred_dir = r'ckpt+ISTD/BDRAR+EMA+U2-net-inp/Best/best_net'
true_dir = r'Datasets/ISTD_Dataset/test/test_B'
u2net_path = r'u2net.pth'

'''
class ShareArgs:
    args = {
        #'epoch': 0,
        #'n_epochs': 200,
        'iter_num': 3000,
        'train_batch_size': 1,
        'last_iter': 0,
        'lr': 5e-3,
        'lr_decay': 0.9,
        'weight_decay': 5e-4,
        'momentum': 0.9,
        'snapshot': '',
        'size': 416
    }

    # 获取参数字典
    def get_args(self):
        return self.args

    # 一次性更新修改所有参数字典的值
    def set_args(self, args):
        self.args = args

    # 根据索引更新参数字典的值
    def set_args_value(self, key, value):
        self.args[key] = value

    # 获取指定索引的默认参数的值
    def get_args_value(self, key, default_value=None):
        return self.args.get(key, default_value)

    # 判断索引是否在参数字典里面
    def contain_key(self, key):
        return key in self.args.keys()

    # 用于更新字典中的键/值对，可以修改存在的键对应的值，也可以添加新的键/值对到字典中
    def update(self, args):
        self.args.update(args)

    # 添加一个新的参数
    def add_arg(self, key, value):
        self.args[key] = value

    # 删除一个参数
    def remove_arg(self, key):
        del self.args[key]
        
'''