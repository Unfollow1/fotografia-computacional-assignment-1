import rawpy
import numpy as np
import matplotlib.pyplot as plt

# lê o arquivo DNG
with rawpy.imread('data/scene_raw.dng') as raw:
    # extrai os pixels brutos sem nenhum processamento
    bayer = raw.raw_image_visible.copy()
    
    # algumas informações úteis
    print("Padrão Bayer:", raw.color_desc)
    print("Dimensões:", bayer.shape)
    print("Valor mínimo:", bayer.min())
    print("Valor máximo:", bayer.max())

plt.imshow(bayer, cmap='gray')
plt.title('Imagem RAW bruta')
plt.savefig('output/01_raw_bruta.png')
plt.show()

# normaliza para o intervalo 0.0 a 1.0
bayer_norm = bayer.astype(np.float64)
bayer_norm = (bayer_norm - bayer_norm.min()) / (bayer_norm.max() - bayer_norm.min())

print("Valor mínimo após normalização:", bayer_norm.min())
print("Valor máximo após normalização:", bayer_norm.max())

plt.imshow(bayer_norm, cmap='gray')
plt.title('Imagem RAW normalizada')
plt.savefig('output/02_raw_normalizada.png')
plt.show()

# demosaicking
altura, largura = bayer_norm.shape

# criando três canais vazios (R, G, B)
R = np.zeros((altura, largura))
G = np.zeros((altura, largura))
B = np.zeros((altura, largura))

# preenche cada canal com os valores que a câmera capturou
# com o padrão RGBG:
# linha par,   coluna par   = R
# linha par,   coluna impar = G
# linha impar, coluna par   = G
# linha impar, coluna impar = B
R[0::2, 0::2] = bayer_norm[0::2, 0::2]
G[0::2, 1::2] = bayer_norm[0::2, 1::2]
G[1::2, 0::2] = bayer_norm[1::2, 0::2]
B[1::2, 1::2] = bayer_norm[1::2, 1::2]

print("canais R, G, B separados.")

# interpolação bilinear
# Usamos uma máscara (kernel) de convolução para calcular a média dos vizinhos
from scipy.ndimage import convolve

# Kernel para interpolar R e B (média das diagonais e vizinhos diretos)
kernel_RB = np.array([ [1/4, 1/2, 1/4],
                       [1/2,  1 , 1/2],
                       [1/4, 1/2, 1/4]])

# Kernel para interpolar G (média dos vizinhos diretos)
kernel_G = np.array([ [0  ,  1/4 , 0  ],
                      [1/4,  1   , 1/4],
                      [0  ,  1/4 , 0  ]])

R = convolve(R, kernel_RB)
G = convolve(G, kernel_G)
B = convolve(B, kernel_RB)

# junta os três canais em uma imagem colorida
imagem_rgb = np.stack([R, G, B], axis=2)
imagem_rgb = np.clip(imagem_rgb, 0, 1)

plt.imshow(imagem_rgb)
plt.title('após demosaicking')
plt.savefig('output/03_demosaicking.png')
plt.show()

print("demosaicking concluído.")

# white balance
# lê os coeficientes gravados pela câmera no arquivo DNG
with rawpy.imread('data/scene_raw.dng') as raw:
    wb = raw.camera_whitebalance
    print("coeficientes de balanço de branco:", wb)

# normaliza os coeficientes pelo verde (mais estável)
wb_r = wb[0] / wb[1]
wb_g = wb[1] / wb[1]
wb_b = wb[2] / wb[1]

print(f"wb_r: {wb_r:.4f}, wb_g: {wb_g:.4f}, wb_b: {wb_b:.4f}")

# multiplica cada canal pelo seu coeficiente wb
R_wb = np.clip(R * wb_r, 0, 1)
G_wb = np.clip(G * wb_g, 0, 1)
B_wb = np.clip(B * wb_b, 0, 1)

# juntando os canais
imagem_wb = np.stack([R_wb, G_wb, B_wb], axis=2)

plt.imshow(imagem_wb)
plt.title('após balanço de branco')
plt.savefig('output/04_balanco_branco.png')
plt.show()

print("balanço de branco concluído!")

# correção gamma
gamma = 2.2
imagem_gamma = np.power(imagem_wb, 1/gamma)

plt.imshow(imagem_gamma)
plt.title('após correção gamma')
plt.savefig('output/05_gamma.png')
plt.show()

print("correção gamma concluída!")