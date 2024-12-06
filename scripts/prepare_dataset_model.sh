# download dataset
gdown https://drive.google.com/uc?id=10lNMVTCVdkQ3lJ5Cc8NNUrS785qXtZCN
unzip datasets.zip
rm datasets.zip

# download pre-computed results
gdown https://drive.google.com/uc?id=10uoJ7WyKHK_apHcTLdJYpSl4_BsVsDva
unzip results.zip
rm results.zip

# download our trained model and animatediff pretrained motion module
mkdir experiments/
cd experiments/
gdown https://drive.google.com/uc?id=19F8OIICfLnbotfg6mivRrHh_-i9G8s1y
unzip pretrained_models.zip
rm pretrained_models.zip

# download stable diffusion foundation model: chilloumix
cd pretrained_models/
git-lfs clone https://huggingface.co/stablediffusionapi/chilloutmix.git
