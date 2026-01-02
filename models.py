import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from torch import Tensor
from torchvision import models

# =============================================================================
# 1. 이미지 인코더 모델 정의
# =============================================================================
class SqueezeExcitation(nn.Module):
    """EfficientNet의 SE 블록 구현"""
    def __init__(self, input_channels, squeezed_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(input_channels, squeezed_channels, 1)
        self.fc2 = nn.Conv2d(squeezed_channels, input_channels, 1)
        self.act = nn.SiLU(inplace=True)
        self.scale_act = nn.Sigmoid()

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.scale_act(scale)
        return x * scale

class MBConvBlock(nn.Module):
    """EfficientNet의 Inverted Residual Block (MBConv) 구현"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        expanded_channels = in_channels * expand_ratio

        # EfficientNet의 기본 BN 파라미터 (eps=1e-3, momentum=0.01)
        bn_eps = 1e-3
        bn_momentum = 0.01

        layers = []
        # 1. Expansion Phase (expand_ratio가 1이면 생략)
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, 1, bias=False))
            layers.append(nn.BatchNorm2d(expanded_channels, eps=bn_eps, momentum=bn_momentum))
            layers.append(nn.SiLU(inplace=True))

        # 2. Depthwise Convolution
        layers.append(nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride,
                                padding=kernel_size//2, groups=expanded_channels, bias=False))
        layers.append(nn.BatchNorm2d(expanded_channels, eps=bn_eps, momentum=bn_momentum))
        layers.append(nn.SiLU(inplace=True))

        # 3. Squeeze and Excitation
        num_squeezed_channels = max(1, int(in_channels * se_ratio))
        layers.append(SqueezeExcitation(expanded_channels, num_squeezed_channels))

        # 4. Pointwise Convolution
        layers.append(nn.Conv2d(expanded_channels, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class CnnFeatureExtractor(nn.Module):
    """
    다양한 CNN 아키텍처의 앞부분을 특징 추출기로 사용하는 범용 클래스입니다.
    config.yaml의 `cnn_feature_extractor.name` 설정에 따라 모델 구조가 결정됩니다.
    """
    def __init__(self, cnn_feature_extractor_name='resnet18_layer1', pretrained=True, featured_patch_dim=None):
        super().__init__()
        self.cnn_feature_extractor_name = cnn_feature_extractor_name

        # CNN 모델 이름에 따라 모델과 잘라낼 레이어, 기본 출력 채널을 설정합니다.
        if cnn_feature_extractor_name == 'resnet18_layer1':
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.conv_front = nn.Sequential(*list(base_model.children())[:5]) # layer1까지
            base_out_channels = 64
        elif cnn_feature_extractor_name == 'resnet18_layer2':
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.conv_front = nn.Sequential(*list(base_model.children())[:6]) # layer2까지
            base_out_channels = 128
            
        elif cnn_feature_extractor_name == 'mobilenet_v3_small_feat1':
            base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self.conv_front = base_model.features[:2] # features의 2번째 블록까지
            base_out_channels = 16
        elif cnn_feature_extractor_name == 'mobilenet_v3_small_feat3':
            base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self.conv_front = base_model.features[:4] # features의 4번째 블록까지
            base_out_channels = 24
        elif cnn_feature_extractor_name == 'mobilenet_v3_small_feat4':
            base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self.conv_front = base_model.features[:5] # features의 5번째 블록까지
            base_out_channels = 40
            
        elif cnn_feature_extractor_name == 'efficientnet_b0_feat2':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self.conv_front = base_model.features[:3] # features의 3번째 블록까지
            base_out_channels = 24
        elif cnn_feature_extractor_name == 'efficientnet_b0_feat3':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self.conv_front = base_model.features[:4] # features의 4번째 블록까지
            base_out_channels = 40
        # --- MobileNetV4 (timm) ---
        elif cnn_feature_extractor_name == 'mobilenet_v4_feat1':
            base_model = timm.create_model('mobilenetv4_conv_small', pretrained=pretrained, features_only=True, out_indices=(0,))
            self.conv_front = base_model
            base_out_channels = 32 # feat1 출력 채널
        elif cnn_feature_extractor_name == 'mobilenet_v4_feat2':
            base_model = timm.create_model('mobilenetv4_conv_small', pretrained=pretrained, features_only=True, out_indices=(0, 1))
            self.conv_front = base_model
            base_out_channels = 48 # feat2 출력 채널
        elif cnn_feature_extractor_name == 'mobilenet_v4_feat3':
            base_model = timm.create_model('mobilenetv4_conv_small', pretrained=pretrained, features_only=True, out_indices=(0, 1, 2))
            self.conv_front = base_model
            base_out_channels = 64 # feat3 출력 채널
        elif cnn_feature_extractor_name == 'mobilenet_v4_feat4':
            base_model = timm.create_model('mobilenetv4_conv_small', pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
            self.conv_front = base_model
            base_out_channels = 96 # feat4 출력 채널
            
        elif cnn_feature_extractor_name == 'custom':
            # EfficientNet-B0 feat2 구조를 직접 코드로 구현 (커스터마이징 용도)
            bn_eps = 1e-5
            bn_momentum = 0.1
            layers = [
                # Stem
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32, eps=bn_eps, momentum=bn_momentum),
                nn.SiLU(inplace=True),
                # Block 1
                MBConvBlock(32, 16, kernel_size=3, stride=1, expand_ratio=1, se_ratio=0.25),
                # Block 2
                MBConvBlock(16, 24, kernel_size=3, stride=2, expand_ratio=6, se_ratio=0.25), 
                MBConvBlock(24, 24, kernel_size=3, stride=1, expand_ratio=6, se_ratio=0.25)
            ]
            self.conv_front = nn.Sequential(*layers)
            base_out_channels = 24      

        else:
            raise ValueError(f"지원하지 않는 CNN 피처 추출기 이름입니다: {cnn_feature_extractor_name}")

        if featured_patch_dim is not None and featured_patch_dim != base_out_channels:
            self.conv_1x1 = nn.Conv2d(base_out_channels, featured_patch_dim, kernel_size=1)
        else:
            self.conv_1x1 = nn.Identity()

    def forward(self, x):
        x = self.conv_front(x)
        x = self.conv_1x1(x) 
        if isinstance(x, list):
            x = x[-1]
        return x

class PatchConvEncoder(nn.Module):
    """
    이미지를 패치로 나누고, 각 패치에서 특징을 추출하여 1D 시퀀스로 변환하는 인코더입니다.
    [수정] 각 패치를 NxN 하위 토큰으로 분할하여, 어텐션이 더 세분화된 단위에서 수행되도록 합니다. (N=pool_dim)
    [ONNX FIX] ONNX 변환 호환성을 위해 kernel size 계산 시 int() 캐스팅을 명시적으로 수행합니다.
    """
    def __init__(self, img_size, patch_size, stride, featured_patch_dim, cnn_feature_extractor_name, pre_trained=True, pool_dim=1, use_token_mixer=True):
        super(PatchConvEncoder, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.featured_patch_dim = featured_patch_dim
        self.pool_dim = pool_dim

        # 원본 패치 그리드 크기
        self.num_patches_H_orig = (img_size - self.patch_size) // self.stride + 1
        self.num_patches_W_orig = (img_size - self.patch_size) // self.stride + 1
        
        # 풀링을 통해 생성된 하위 토큰을 포함한 새로운 그리드 크기 및 토큰 수
        self.grid_size_h = self.num_patches_H_orig * self.pool_dim
        self.grid_size_w = self.num_patches_W_orig * self.pool_dim
        self.num_encoder_patches = self.grid_size_h * self.grid_size_w

        # 1. CNN Feature Extractor
        self.feature_extractor = CnnFeatureExtractor(
            cnn_feature_extractor_name=cnn_feature_extractor_name,
            pretrained=pre_trained,
            featured_patch_dim=featured_patch_dim
        )
        
        # 3. Token Mixer
        if use_token_mixer:
            self.token_mixer = nn.Sequential(
                nn.Conv2d(featured_patch_dim, featured_patch_dim, kernel_size=3, padding=1, groups=featured_patch_dim, bias=False),
                nn.BatchNorm2d(featured_patch_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.token_mixer = nn.Identity()

        self.norm = nn.LayerNorm(featured_patch_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 이미지를 원본 패치로 분할
        patches = x.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(-1, C, self.patch_size, self.patch_size) # [B*N_orig, C, P, P]

        # 2. 각 패치별 특징 추출 및 풀링
        patch_features = self.feature_extractor(patches)       # [B*N_orig, D, Hf, Wf]
        
        # [ONNX FIX 2] 커널 크기 계산 시 Tensor가 아닌 순수 int형으로 변환
        # ONNX Tracing 과정에서 shape가 Tensor로 잡혀 avg_pool2d의 인자 타입 에러가 나는 것을 방지합니다.
        h_feat = int(patch_features.size(2))
        w_feat = int(patch_features.size(3))
        
        kernel_h = int(h_feat // self.pool_dim)
        kernel_w = int(w_feat // self.pool_dim)
        
        pooled_features = F.avg_pool2d(
            patch_features, 
            kernel_size=(kernel_h, kernel_w), 
            stride=(kernel_h, kernel_w)
        )
        # pooled_features: [B*N_orig, D, pool_dim, pool_dim]
        
        # 3. 하위 토큰들을 2D 그리드로 재구성
        tokens_as_grid = pooled_features.view(
            B, self.num_patches_H_orig, self.num_patches_W_orig, self.featured_patch_dim, self.pool_dim, self.pool_dim
        )
        tokens_as_grid = tokens_as_grid.permute(0, 3, 1, 4, 2, 5)
        tokens_as_grid = tokens_as_grid.reshape(B, self.featured_patch_dim, self.grid_size_h, self.grid_size_w)

        # 4. 세분화된 그리드 위에서 Token Mixing 수행
        mixed_grid = self.token_mixer(tokens_as_grid) # [B, D, H_new, W_new]

        # 5. 디코더로 전달할 최종 토큰 시퀀스로 변환
        mixed_tokens = mixed_grid.permute(0, 2, 3, 1).contiguous().view(B, -1, self.featured_patch_dim)

        final_tokens = self.norm(mixed_tokens)
        return final_tokens

# =============================================================================
# 2. 디코더 모델 정의
# =============================================================================

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class Embedding4Decoder(nn.Module):
    """
    Decoder 입력을 위한 임베딩 레이어.
    """
    def __init__(self, num_encoder_patches, featured_patch_dim, num_decoder_patches,
                 adaptive_initial_query=False, num_decoder_layers=3, emb_dim=128, num_heads=16,
                 decoder_ff_dim=256, attn_dropout=0., dropout=0., save_attention=False, res_attention=False,
                 positional_encoding=True, pos_encoding_type="2d"):

        super().__init__()

        self.adaptive_initial_query = adaptive_initial_query
        self.emb_dim = emb_dim
        self.pos_encoding_type = pos_encoding_type

        # --- 입력 인코딩 ---
        self.W_feat2emb = nn.Linear(featured_patch_dim, emb_dim)
        self.W_Q_init = nn.Linear(featured_patch_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        # --- 학습 가능한 쿼리 ---
        self.learnable_queries = nn.Parameter(torch.empty(num_decoder_patches, featured_patch_dim))
        nn.init.xavier_uniform_(self.learnable_queries)

        # --- Positional Encoding ---
        self.use_positional_encoding = positional_encoding
        self.pos_embed = None
        self._pe_cache_key = None

        if self.adaptive_initial_query:
            self.W_K_init = nn.Linear(emb_dim, emb_dim)
            self.W_V_init = nn.Linear(emb_dim, emb_dim)

        # --- 디코더 ---
        self.decoder = Decoder(
            num_encoder_patches, emb_dim, num_heads, num_decoder_patches,
            decoder_ff_dim=decoder_ff_dim, attn_dropout=attn_dropout, dropout=dropout,
            res_attention=res_attention, num_decoder_layers=num_decoder_layers, save_attention=save_attention
        )

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        omega = torch.arange(embed_dim // 2, dtype=torch.float, device=pos.device)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega 

        pos = pos.reshape(-1) 
        out = torch.einsum('m,d->md', pos, omega) 

        emb_sin = torch.sin(out) 
        emb_cos = torch.cos(out) 

        emb = torch.cat([emb_sin, emb_cos], dim=1) 
        return emb

    def get_2d_sincos_pos_embed(self, embed_dim, grid_h, grid_w, device=None):
        assert embed_dim % 2 == 0
        device = device or torch.device('cpu')

        grid_h_arange = torch.arange(grid_h, dtype=torch.float32, device=device)
        grid_w_arange = torch.arange(grid_w, dtype=torch.float32, device=device)

        grid_h_coords, grid_w_coords = torch.meshgrid(grid_h_arange, grid_w_arange, indexing='ij')

        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h_coords) 
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w_coords) 

        pos_embed = torch.cat([emb_h, emb_w], dim=1) 
        return pos_embed.unsqueeze(0) 

    def get_polar_sincos_pos_embed(self, embed_dim, grid_h, grid_w, device=None, eps=1e-6):
        assert embed_dim % 4 == 0
        device = device or torch.device('cpu')

        ys = torch.linspace(-1.0, 1.0, steps=grid_h, device=device)
        xs = torch.linspace(-1.0, 1.0, steps=grid_w, device=device)

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij') 
        x = grid_x.reshape(-1) 
        y = grid_y.reshape(-1) 

        r = torch.sqrt(x * x + y * y)
        r = r / (r.max() + eps) 
        theta = torch.atan2(y, x) / math.pi 

        emb_r = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, r) 
        emb_t = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, theta) 

        pos = torch.cat([emb_r, emb_t], dim=1) 
        return pos.unsqueeze(0) 

    def forward(self, x, grid_h=None, grid_w=None) -> Tensor:
        bs = x.shape[0]

        x = self.W_feat2emb(x) 
        x_clean = self.dropout(x)

        if self.use_positional_encoding:
            if grid_h is None or grid_w is None:
                raise ValueError("Dynamic PE generation requires grid_h and grid_w.")

            # ONNX Tracing 시 상수화 보장
            grid_h = int(grid_h) if isinstance(grid_h, Tensor) else grid_h
            grid_w = int(grid_w) if isinstance(grid_w, Tensor) else grid_w

            key = (grid_h, grid_w, x.device, x.dtype, str(self.pos_encoding_type))

            if (self.pos_embed is None) or (self._pe_cache_key != key) or (self.pos_embed.shape[1] != x.shape[1]):
                if self.pos_encoding_type == "polar":
                    pe = self.get_polar_sincos_pos_embed(self.emb_dim, grid_h, grid_w, device=x.device)
                else: 
                    pe = self.get_2d_sincos_pos_embed(self.emb_dim, grid_h, grid_w, device=x.device)

                self.pos_embed = pe.to(device=x.device, dtype=x.dtype)
                self._pe_cache_key = key

            x = x + self.pos_embed

        seq_encoder_patches = self.dropout(x)

        if self.adaptive_initial_query:
            latent_queries = self.W_Q_init(self.learnable_queries)
            latent_queries = latent_queries.unsqueeze(0).expand(bs, -1, -1)

            k_init = self.W_K_init(seq_encoder_patches)
            v_init = self.W_V_init(seq_encoder_patches) 

            latent_attn_scores = torch.bmm(latent_queries, k_init.transpose(1, 2))
            latent_attn_weights = F.softmax(latent_attn_scores, dim=-1)

            seq_decoder_patches = torch.bmm(latent_attn_weights, v_init)
        else:
            learnable_queries = self.W_Q_init(self.learnable_queries)
            latent_queries = latent_queries.unsqueeze(0).expand(bs, -1, -1)

        return seq_encoder_patches, seq_decoder_patches
            

class Projection4Classifier(nn.Module):
    def __init__(self, emb_dim, featured_patch_dim):
        super().__init__()
        self.linear = nn.Linear(emb_dim, featured_patch_dim)
        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, x):
        x = self.linear(x)
        x = self.flatten(x)
        return x 
            
class Decoder(nn.Module):
    def __init__(self, num_encoder_patches, emb_dim, num_heads, num_decoder_patches, decoder_ff_dim=None, attn_dropout=0., dropout=0.,
                    res_attention=False, num_decoder_layers=1, save_attention=False):
        super().__init__()
        
        self.layers = nn.ModuleList([DecoderLayer(num_encoder_patches, emb_dim, num_decoder_patches, num_heads=num_heads, decoder_ff_dim=decoder_ff_dim, attn_dropout=attn_dropout, dropout=dropout,
                                                        res_attention=res_attention, save_attention=save_attention) for i in range(num_decoder_layers)])
        self.res_attention = res_attention

    def forward(self, seq_encoder:Tensor, seq_decoder:Tensor):
        scores = None
        if self.res_attention:
            for mod in self.layers: _, seq_decoder, scores = mod(seq_encoder, seq_decoder, prev=scores)
            return seq_decoder
        else:
            for mod in self.layers: _, seq_decoder = mod(seq_encoder, seq_decoder)
            return seq_decoder

class DecoderLayer(nn.Module):
    def __init__(self, num_encoder_patches, emb_dim, num_decoder_patches, num_heads, decoder_ff_dim=256, save_attention=False,
                    attn_dropout=0, dropout=0., bias=True, res_attention=False):
        super().__init__()
        assert not emb_dim%num_heads, f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})"
        
        self.res_attention = res_attention
        self.cross_attn = _MultiheadAttention(emb_dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention, qkv_bias=True)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(emb_dim)

        self.ffn = nn.Sequential(nn.Linear(emb_dim, decoder_ff_dim, bias=bias),
                                GEGLU(),
                                nn.Dropout(dropout),
                                nn.Linear(decoder_ff_dim//2, emb_dim, bias=bias)) 
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(emb_dim)
        
        self.save_attention = save_attention

    def forward(self, seq_encoder:Tensor, seq_decoder:Tensor, prev=None) -> Tensor:
        if self.res_attention:
            decoder_out, attn, scores = self.cross_attn(seq_decoder, seq_encoder, seq_encoder, prev)
        else:
            decoder_out, attn = self.cross_attn(seq_decoder, seq_encoder, seq_encoder)
        
        if self.save_attention:
            self.attn = attn
        
        seq_decoder = seq_decoder + self.dropout_attn(decoder_out)
        seq_decoder = self.norm_attn(seq_decoder)
        
        ffn_out = self.ffn(seq_decoder)
        seq_decoder = seq_decoder + self.dropout_ffn(ffn_out)  
        seq_decoder = self.norm_ffn(seq_decoder)
        
        if self.res_attention: return seq_encoder, seq_decoder, scores
        else: return seq_encoder, seq_decoder

class _MultiheadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, **kwargs):
        super().__init__()
        
        head_dim = emb_dim // num_heads
        self.scale = head_dim**-0.5
        self.num_heads, self.head_dim = num_heads, head_dim

        self.W_Q = nn.Linear(emb_dim, head_dim * num_heads, bias=qkv_bias)
        self.W_K = nn.Linear(emb_dim, head_dim * num_heads, bias=qkv_bias)
        self.W_V = nn.Linear(emb_dim, head_dim * num_heads, bias=qkv_bias)

        self.res_attention = res_attention
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.concatheads2emb = nn.Sequential(nn.Linear(num_heads * head_dim, emb_dim), nn.Dropout(proj_dropout))

    def forward(self, Q:Tensor, K:Tensor, V:Tensor, prev=None):
        bs = Q.size(0)
        
        q_s = self.W_Q(Q).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_s = self.W_K(K).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_s = self.W_V(V).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q_s, k_s.transpose(-2, -1)) * self.scale
        
        if prev is not None: attn_scores = attn_scores + prev
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v_s)
        output = output.permute(0, 2, 1, 3).contiguous().view(bs, -1, self.num_heads * self.head_dim)
        
        output = self.concatheads2emb(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class Model(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        
        num_encoder_patches = args.num_encoder_patches 
        num_labels = args.num_labels 
        num_decoder_patches = args.num_decoder_patches 
        self.featured_patch_dim = args.featured_patch_dim 
        adaptive_initial_query = args.adaptive_initial_query 
        emb_dim = args.emb_dim           
        num_heads = args.num_heads           
        num_decoder_layers = args.num_decoder_layers 
        decoder_ff_ratio = args.decoder_ff_ratio 
        dropout = args.dropout           
        attn_dropout = dropout           
        positional_encoding = args.positional_encoding 
        save_attention = args.save_attention     
        res_attention = getattr(args, 'res_attention', False)
        pos_encoding_type = getattr(args, 'pos_encoding_type', '2d')

        decoder_ff_dim = emb_dim * decoder_ff_ratio 

        self.embedding4decoder = Embedding4Decoder(
            num_encoder_patches=num_encoder_patches,
            featured_patch_dim=self.featured_patch_dim,
            num_decoder_patches=num_decoder_patches,
            adaptive_initial_query=adaptive_initial_query,
            num_decoder_layers=num_decoder_layers,
            emb_dim=emb_dim,
            num_heads=num_heads,
            decoder_ff_dim=decoder_ff_dim,
            positional_encoding=positional_encoding,
            pos_encoding_type=pos_encoding_type,
            attn_dropout=attn_dropout,
            dropout=dropout,
            res_attention=res_attention,
            save_attention=save_attention
        )

        self.projection4classifier = Projection4Classifier(emb_dim, self.featured_patch_dim)

    def forward(self, x, grid_h=None, grid_w=None):
        seq_encoder_patches, seq_decoder_patches = self.embedding4decoder(x, grid_h=grid_h, grid_w=grid_w)
        z = self.embedding4decoder.decoder(seq_encoder_patches, seq_decoder_patches)
        features = self.projection4classifier(z)
        return features

# =============================================================================
# 3. 전체 모델 구성
# =============================================================================
class Classifier(nn.Module):
    def __init__(self, num_decoder_patches, featured_patch_dim, num_labels, dropout):
        super().__init__()
        input_dim = num_decoder_patches * featured_patch_dim 
        hidden_dim = (input_dim + num_labels) // 2 

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x):
        x = self.projection(x) 
        return x

class HybridModel(torch.nn.Module):
    def __init__(self, encoder, decoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        
    def forward(self, x):
        x = self.encoder(x)
        
        # ONNX Tracing 시 grid size가 Tensor로 전파될 수 있으므로 int() 변환
        grid_h = getattr(self.encoder, 'grid_size_h', None)
        grid_w = getattr(self.encoder, 'grid_size_w', None)
        
        # 모델 내부에서 int()로 처리하지만 여기서도 명시적으로 상수로 전달
        if isinstance(grid_h, Tensor): grid_h = int(grid_h)
        if isinstance(grid_w, Tensor): grid_w = int(grid_w)

        x = self.decoder(x, grid_h=grid_h, grid_w=grid_w)

        out = self.classifier(x)
        return out