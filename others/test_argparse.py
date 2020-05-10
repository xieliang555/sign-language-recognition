



def test(device, resume_training):
    print(device)
    print(resume_training)
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="0",
        type=str, help='the index of device')
    parser.add_argument('--resume_training', action='store_true',
        help='If action, resume_training')
    
    args = parser.parse_args()
    device = args.device
    resume_training = args.resume_training
    
    test(device, resume_training)