args = dict()


#project structure
args["CODE_DIRECTORY"] = "/home/python/deep_avsr_conformer"  #absolute path to the code directory
args["DATA_DIRECTORY"] = "/home/python/datasets/mvlrs_v1"  #absolute path to the data directory
args["NOISE_DIRECTORY"] = "/home/python/datasets/JNASNOISE_16000" #absolute path to the noise directory
args["DEMO_DIRECTORY"] = None   #absolute path to the demo directory
args["PRETRAINED_MODEL_FILE"] = "/audio_visual/record/checkpoints_VO21/models/pretrain_021w-step_0100-wer_0.950.pt"     #relative path to the pretrained model file "/audio_visual/..."
args["TRAINED_MODEL_FILE"] = "/audio_visual/record/checkpoints_AV_train_joint256/models/train-step_0300-wer_0.198.pt"   #relative path to the trained model file
# args["TRAINED_MODEL_FILE"] = "/audio_visual/record/checkpoints_AV_train_v2/models/train-step_0300-wer_0.221.pt"   #relative path to the trained model file
# args["TRAINED_MODEL_FILE"] = "/audio_visual/record/checkpoints_AOtrain/models/train-step_0120-wer_0.268.pt"   #relative path to the trained model file
args["TRAINED_LM_FILE"] = "/home/python/deep_avsr_conformer/audio_visual/final/models/language_model.pt"  #absolute path to the trained language model file
args["TRAINED_FRONTEND_FILE"] = "/home/python/deep_avsr_conformer/audio_visual/final/models/lrw_resnet18_mstcn_adamw_s3.pth.tar" #absolute path to the trained visual frontend file


#data
args["PRETRAIN_VAL_SPLIT"] = 0.01   #validation set size fraction during pretraining
args["NUM_WORKERS"] = 20 #dataloader num_workers argument
args["PRETRAIN_NUM_WORDS"] = 29  #number of words limit in current curriculum learning iteration
args["MAIN_REQ_INPUT_LENGTH"] = 145 #minimum input length while training
args["CHAR_TO_INDEX"] = {" ":1, "'":22, "1":30, "0":29, "3":37, "2":32, "5":34, "4":38, "7":36, "6":35, "9":31, "8":33,
                         "A":5, "C":17, "B":20, "E":2, "D":12, "G":16, "F":19, "I":6, "H":9, "K":24, "J":25, "M":18,
                         "L":11, "O":4, "N":7, "Q":27, "P":21, "S":8, "R":10, "U":13, "T":3, "W":15, "V":23, "Y":14,
                         "X":26, "Z":28, "<EOS>":39, "<BOS>":40}    #character to index mapping
args["INDEX_TO_CHAR"] = {1:" ", 22:"'", 30:"1", 29:"0", 37:"3", 32:"2", 34:"5", 38:"4", 36:"7", 35:"6", 31:"9", 33:"8",
                         5:"A", 17:"C", 20:"B", 2:"E", 12:"D", 16:"G", 19:"F", 6:"I", 9:"H", 24:"K", 25:"J", 18:"M",
                         11:"L", 4:"O", 7:"N", 27:"Q", 21:"P", 8:"S", 10:"R", 13:"U", 3:"T", 15:"W", 23:"V", 14:"Y",
                         26:"X", 28:"Z", 39:"<EOS>", 40:"<BOS>"}    #index to character reverse mapping
args["lambda"] = 0.9

#audio preprocessing
args["NOISE_PROBABILITY"] = 0.5    #noise addition probability while training
args["NOISE_SNR_DB"] = [-5]    #noise level in dB SNR
args["STFT_WINDOW"] = "hamming" #window to use while computing STFT
args["STFT_WIN_LENGTH"] = 0.040 #window size in secs for computing STFT
args["STFT_OVERLAP"] = 0.030    #consecutive window overlap in secs while computing STFT

# specaugment
args["F_SIZE"] = 27 # size of the feature mask
args["T_RATIO"] = 0.1 # ratio of the time mask
args["M_F"] = 2 # number of the feature mask
args["M_T"] = 2 # number of ther time mask


#video preprocessing
args["VIDEO_FPS"] = 25  #frame rate of the video clips
args["ROI_SIZE"] = 96  #height and width of input greyscale lip region patch
args["NORMALIZATION_MEAN"] = 0.4161 #mean value for normalization of greyscale lip region patch
args["NORMALIZATION_STD"] = 0.1688  #standard deviation value for normalization of greyscale lip region patch


#training
args["SEED"] = 324 #seed for random number generators
args["BATCH_SIZE"] = 20 #minibatch size
args["STEP_SIZE"] = 16384   #number of samples in one step (virtual epoch)
args["NUM_STEPS"] = 1000 #maximum number of steps to train for (early stopping is used)
args["SAVE_FREQUENCY"] = 10 #saving the model weights and loss/metric plots after every these many steps


#optimizer, scheduler and modality dropping
args["INIT_LR"] = 1e-4  #initial learning rate for scheduler
args["FINAL_LR"] = 1e-6 #final learning rate for scheduler
args["LR_SCHEDULER_FACTOR"] = 0.5   #learning rate decrease factor for scheduler
args["LR_SCHEDULER_WAIT"] = 10  #number of steps to wait to lower learning rate
args["LR_SCHEDULER_THRESH"] = 0.001 #threshold to check plateau-ing of wer
args["MOMENTUM1"] = 0.9 #optimizer momentum 1 value
args["MOMENTUM2"] = 0.999   #optimizer momentum 2 value
args["AUDIO_ONLY_PROBABILITY"] = 0    #probability with which to use only the audio modality
args["VIDEO_ONLY_PROBABILITY"] = 1    #probability with which to use only the video modality


#model
args["AUDIO_FEATURE_SIZE"] = 80    #feature size of audio features
args["NUM_CLASSES"] = 40    #number of output characters


#transformer architecture
args["PE_MAX_LENGTH"] = 2500    #length up to which we calculate positional encodings
args["TX_NUM_FEATURES"] = 256   #transformer input feature size
args["TX_ATTENTION_HEADS"] = 8  #number of attention heads in multihead attention layer
args["TX_NUM_LAYERS"] = 12   #number of Transformer Encoder blocks in the stack
args["TX_FEEDFORWARD_DIM"] = 2048   #hidden layer size in feedforward network of transformer
args["TX_DROPOUT"] = 0.1    #dropout probability in the transformer


#beam search
args["BEAM_WIDTH"] = 20    #beam width
args["LM_WEIGHT_ALPHA"] = 0.5   #weight of language model probability in shallow fusion beam scoring
args["LENGTH_PENALTY_BETA"] = 0.1   #length penalty exponent hyperparameter
args["THRESH_PROBABILITY"] = 0.0001 #threshold probability in beam search algorithm
args["USE_LM"] = False  #whether to use language model for decoding


#testing
args["TEST_DEMO_DECODING"] = "search"   #test/demo decoding type - "greedy" or "search". (pretrain and train are "greedy")
args["TEST_DEMO_NOISY"] = True #test/demo with noisy audio
args["TEST_DEMO_MODE"] = "AV"   #mode to use AV model in - "AO" or "VO" or "AV"


if __name__ == "__main__":

    for key,value in args.items():
        print(str(key) + " : " + str(value))
