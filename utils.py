import logging, time, os, torch

def create_logger(logger_file_path):
    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(logger_file_path, log_name)

    logger = logging.getLogger() #创建logger对象
    logger.setLevel(logging.INFO) #设置日志级别

    file_handler = logging.FileHandler(final_log_file) #创建文件处理器
    console_handler = logging.StreamHandler() #创建控制台处理器

    #输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'], False)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
        return start_epoch
    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0 
    

def save_checkpoint(model, optimizer, epoch, name):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint_dir = f'{name}_checkpoints/Epoch_{epoch}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}.")


