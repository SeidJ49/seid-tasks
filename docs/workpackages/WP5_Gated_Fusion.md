## Work Package 5: Reliability-Guided Gated Fusion – Detaillierte Beschreibung

Nach WP4 (Reliability-Map) geht es in WP5 um die Integration eines leichtgewichtigen Gate-Mechanismus, der LiDAR und Radar adaptiv fusioniert.

---

### 1. Ziel von WP5

Ein **leichtgewichtiges Fusions-Gate** implementieren, das:

1. **LiDAR und Radar BEV-Features** basierend auf der Reliability-Map (`U_L`) fusioniert
2. **Räumlich variabel** ist (Gate pro BEV-Zelle, nicht global)
3. **Einfach zu interpretieren** ist (Gewichtsmap, kein Black-Box-Lernen)
4. **Robustheit unter schlechtem Wetter** verbessert

---

### 2. Wichtige Erwartung: Spatial Gate, nicht Channel-wise

### Warum Spatial Gate (zuerst)?

| Eigenschaft | Spatial Gate | Channel-wise Gate |
|-------------|--------------|-------------------|
| **Debugbarkeit** | + Einfach zu visualisieren | - Komplexer |
| **Interpretierbarkeit** | + Jede Zelle hat ein Gewicht | - 128 Gewichte pro Zelle |
| **Erste Validierung** | + Ausreichend für Konzept | - Overkill für Baseline |

**Empfehlung:** Starte mit **Spatial Gate**. Channel-weise Gating ist später eine optionale Verbesserung (Ablation).

---

### 3. Konkrete technische Schritte

#### Schritt 1: Gate-Formel

$$  
W = \sigma(\phi([F_L, F_R, U_L]))  
$$

wobei:

- $F_L$ = LiDAR BEV-Features, Shape: `(B, C_L, H, W)`
- $F_R$ = Radar BEV-Features, Shape: `(B, C_R, H, W)` (gleiche räumliche Größe wie $F_L$)
- $U_L$ = LiDAR Reliability-Map aus WP4, Shape: `(B, 1, H, W)`
- $\phi$ = kleine Convolutional Block
- $\sigma$ = Sigmoid-Aktivierung

**Ausgabe:**
- $W$ = Spatial Fusion Gate, Shape: `(B, 1, H, W)`, Werte in [0, 1]

#### Schritt 2: Gate-Block-Design ($\phi$)

Implementiere einen einfachen Convolutional Block:

```python
class GateBlock(nn.Module):
    def __init__(self, input_channels, output_channels=1):
        super().__init__()
        # Input hat 2*C + 1 Kanäle (F_L + F_R + U_L)
        self.conv1 = nn.Conv2d(input_channels, input_channels // 4, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(input_channels // 4, output_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)  # Output in [0, 1]
        return x
```

**Design-Rationale:**
- 1×1 Convolutions (keine räumliche Correlation nötig)
- Einfache Expansion: C → C/4 (Dimensionalität reduzieren)
- Sigmoid für Gewichte in [0, 1]

#### Schritt 3: Fusion-Formel

Nach Gate-Berechnung:

$$  
F_{fused} = (1 - W) \odot F_L + W \odot F_R  
$$

Implementierung:

```python
def fused_features(F_L, F_R, W):
    # W: (B, 1, H, W)
    # F_L, F_R: (B, C, H, W)
    F_fused = (1 - W) * F_L + W * F_R
    return F_fused  # Shape: (B, C, H, W)
```

**Interpretierung:**
- $W = 0$ an dieser Zelle → vertraue nur LiDAR
- $W = 1$ an dieser Zelle → vertraue nur Radar
- $W = 0.5$ an dieser Zelle → verwende gleiche Anteile

#### Schritt 4: Integration ins Modell

```python
class LiDARRadarFusionDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lidar_encoder = PointPillarEncoder(...)  # VFE + Scatter + Backbone
        self.radar_encoder = PointPillarEncoder(...)  # VFE + Scatter + Backbone
        self.reliability_head = ReliabilityHead(...)  # Aus WP4
        self.gate_block = GateBlock(...)              # Aus WP5
        self.backbone = BaseBEVBackbone(...)
        self.detection_head = DetectionHead(...)
    
    def forward(self, data_dict):
        # Encode LiDAR und Radar separat
        F_L = self.lidar_encoder(data_dict['processed_lidar'])  # (B, C, H, W)
        F_R = self.radar_encoder(data_dict['processed_radar'])  # (B, C, H, W)
        
        # Compute Reliability Map
        U_L = self.reliability_head(F_L)  # (B, 1, H, W)
        
        # Compute Gate
        concatenated = torch.cat([F_L, F_R, U_L], dim=1)  # (B, 2C+1, H, W)
        W = self.gate_block(concatenated)  # (B, 1, H, W)
        
        # Fuse
        F_fused = (1 - W) * F_L + W * F_R  # (B, C, H, W)
        
        # Shared Backbone + Heads
        features = self.backbone(F_fused)
        output = self.detection_head(features)
        
        return output, W, U_L  # Return gate for visualization
```

---

### 4. Gate-Regularisierung (optional, nur bei Notwendigkeit)

### Wichtige Erwartung

Füge Gate-Regularisierung **nicht** sofort hinzu, es sei denn:

- Gate kollabiert auf 0 (vertraue immer LiDAR)
- Gate kollabiert auf 1 (vertraue immer Radar)
- Unstabile oder verrauschte Switches

### Erste Regularisierung (bei Bedarf): Smoothness Prior

Falls Regularisierung nötig ist:

$$  
\mathcal{L}_{smooth} = \lambda_{smooth} \cdot \left( \sum |\nabla_x W| + |\nabla_y W| \right)  
$$

Implementierung:

```python
def smoothness_loss(gate, lambda_smooth=0.01):
    # Gradient in x-Richtung
    grad_x = gate[:, :, :, 1:] - gate[:, :, :, :-1]
    # Gradient in y-Richtung
    grad_y = gate[:, :, 1:, :] - gate[:, :, :-1, :]
    # L1 Norm
    loss = lambda_smooth * (torch.abs(grad_x).mean() + torch.abs(grad_y).mean())
    return loss
```

**Startparameter:** `lambda_smooth = 0.01` – erhöhe nur bei Bedarf.

---

### 5. Erforderliche Vergleiche (Ablationen)

Die Thesis muss folgende Baselines vergleichen:

| Baseline | Beschreibung |
|----------|-------------|
| **Simple Concat Fusion** | $F_{fused} = [F_L, F_R]$ (no gating) |
| **Gate ohne Reliability** | $W = \sigma(\phi([F_L, F_R]))$ (keine $U_L$) |
| **Gate mit Reliability** | $W = \sigma(\phi([F_L, F_R, U_L]))$ (vollständig) |
| **Naive Fusion (aus WP1)** | Einfaches Average oder Sum |

Erwartung: **Gate mit Reliability > Gate ohne Reliability > Simple Fusion**

---

### 6. Visualisierungen und Analyse

#### Gate-Heatmaps

Erstelle Heatmap-Visualisierungen der Gate-Werte $W$ für verschiedene Wetterbedingungen:

```
Clear Weather:        Light Fog:            Heavy Rain:
[0.3 0.2 0.3]        [0.4 0.5 0.5]         [0.7 0.8 0.8]
[0.2 0.1 0.2]  →     [0.5 0.6 0.5]   →    [0.8 0.9 0.9]
[0.3 0.2 0.3]        [0.4 0.5 0.4]         [0.7 0.8 0.7]
```

Interpretierung:
- Clear: Gate meist **niedrig** (vertraue LiDAR)
- Fog: Gate **moderat** (balance)
- Rain: Gate **hoch** (vertraue mehr Radar)

#### Gate Usage Statistics

Berechne aggregierte Gate-Statistiken pro Wetterbedingung:

| Wetterbedingung | Mean(W) | Std(W) | Max(W) |
|----------------|---------|--------|--------|
| Clear          | 0.25    | 0.15   | 0.60   |
| Light Fog      | 0.45    | 0.20   | 0.80   |
| Heavy Rain     | 0.68    | 0.18   | 0.95   |

**Erwartung:** Mean(W) sollte mit Wetter-Degradation steigen.

---

### 7. Was ist *nicht* Teil von WP5?

- **Keine** Channel-weise Gating (das ist Ablation)
- **Keine** Query-basierte Fusion
- **Keine** Transformer-Cross-Attention
- **Keine** komplexe Gate-Regularisierung (nur wenn nötig)

WP5 ist rein ein **leichtgewichtiges Spatial Gate** mit BEV-Features.

---

### 8. Dokumentationspflichten

1. **Gate-Block-Modul** – Code-Implementation und Design-Rationale
2. **Gate-Heatmaps** – Visualisierungen für Clear/Fog/Rain
3. **Gate Usage Statistics** – Tabelle mit Mean/Std/Max pro Wetter
4. **Ablation-Vergleiche** – Performance mit/ohne Reliability Map
5. **Komparative Analyse** – Warum Gate mit Reliability besser ist

---

### 9. Meilenstein für WP5

**Du bist fertig mit WP5, wenn:**

- [ ] Ein `GateBlock`-Modul implementiert ist
- [ ] Die Fusion-Formel $F_{fused} = (1-W) \odot F_L + W \odot F_R$ integriert ist
- [ ] Gate-Heatmaps für Clear/Fog/Rain generiert sind
- [ ] Gate-Usage-Statistiken berechnet und dokumentiert sind
- [ ] Ablationen durchgeführt: Gate ohne Reliability vs. mit Reliability
- [ ] Kein Gate-Kollaps beobachtet (W nicht nur 0 oder 1)

**Dann** bist du bereit für **WP6 (Training Strategy)**.
