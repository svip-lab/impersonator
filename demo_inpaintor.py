from models.models import ModelsFactory
from options.test_options import TestOptions
from utils.visualizer.demo_visualizer import MotionImitationVisualizer

if __name__ == "__main__":

    opt = TestOptions().parse()

    # set inpaintor
    inpaintor = ModelsFactory.get_by_name(opt.model, opt)

    if opt.visual:
        visualizer = MotionImitationVisualizer(env=opt.name, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    src_path = opt.src_path

    inpaintor.inference(src_path, smpl=None, visualizer=visualizer)