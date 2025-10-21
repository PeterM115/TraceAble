from facenet_pytorch import MTCNN, InceptionResnetV1
import open_clip
    

def load_models():
    mtcnn = MTCNN(image_size=160, margin=14, keep_all=True)
    facenet = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, facenet

# Seperate function as there are many returned values, less messy than grouping with load_models imo.
def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model.eval(), preprocess, tokenizer