## Work Package 6: Training Strategy – Detaillierte Beschreibung

WP6 ist die finale Integration aller Komponenten (WP1–WP5) in ein strukturiertes, gestuftes Trainings-Pipeline.

---

### 1. Ziel von WP6

Ein **strukturiertes, mehrstufiges Trainings-Regime** implementieren, das alle komponenten richtig sequenziert und integriert:

1. **Stage 1:** LiDAR Teacher trainieren auf Clear-Weather
2. **Stage 2:** Radar mit Distillation trainieren (Teacher frozen)
3. **Stage 3:** Vollständiges Fusion-Modell auf Mixed Weather trainieren

---

### 2. Trainings-Architektur

### Stage 1: LiDAR Teacher Training

**Ziel:** Ein starkes LiDAR-Modell auf Clear-Weather-Daten aufbauen.

**Konfiguration:**
- Modell: PointPillar LiDAR-only Encoder → Backbone → Detection Head
- Daten: Clear-Weather Split nur
- Loss: Standard Detection Loss (z.B. PointPillarLoss)
- Dauer: ~100–200 Epochen (bis konvergenz)

**Ausgang:** Trainiertes LiDAR-Teacher-Checkpoint

```python
# Pseudocode für Stage 1
def train_stage1(config):
    model = LiDAROnlyPointPillar(config)
    optimizer = torch.optim.Adam(model.parameters())
    loader = get_clear_weather_loader(config)  # Clear only
    
    for epoch in range(config.epochs):
        for batch in loader:
            output = model(batch)
            loss = detection_loss(output, batch['labels'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Save teacher
    torch.save(model.state_dict(), 'checkpoint_teacher.pth')
    return model
```

**Metriken nach Stage 1:**
- mAP auf Clear-Weather Validation
- mAP auf Fog/Rain (wird wahrscheinlich schlecht sein, das ist ok)

---

### Stage 2: Motion-Aware Radar Distillation Training

**Ziel:** Den Radar-Encoder trainieren mit Motion-Masked Supervision vom gefrorenen LiDAR Teacher.

**Konfiguration:**
- Modell: PointPillar Radar-only Encoder + Distillation Loss
- Daten: Clear-Weather Split nur
- Loss: `L_total = L_detection + lambda * L_distill` (λ = 0.1)
- Teacher: Geladen aus Stage 1 und gefroren (`requires_grad = False`)
- Dauer: ~100–200 Epochen

**Wichtig:** Der Teacher wird **nicht** trainiert, nur der Radar-Encoder.

```python
# Pseudocode für Stage 2
def train_stage2(config):
    # Load and freeze teacher
    teacher = LiDAROnlyPointPillar(config)
    teacher.load_state_dict(torch.load('checkpoint_teacher.pth'))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Create radar encoder
    radar_model = RadarOnlyPointPillar(config)
    optimizer = torch.optim.Adam(radar_model.parameters())
    loader = get_clear_weather_loader(config)  # Clear only
    
    for epoch in range(config.epochs):
        for batch in loader:
            # Get features
            radar_features = radar_model.encoder(batch['processed_radar'])
            with torch.no_grad():
                teacher_features = teacher.encoder(batch['processed_lidar'])
            
            # Compute mask
            mask = get_doppler_mask(batch['radar_doppler'])
            
            # Loss
            det_loss = detection_loss(radar_model.head(radar_features), batch['labels'])
            distill_loss = masked_mse_loss(radar_features, teacher_features, mask)
            total_loss = det_loss + 0.1 * distill_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
    # Save radar encoder
    torch.save(radar_model.state_dict(), 'checkpoint_radar_distilled.pth')
    return radar_model
```

**Metriken nach Stage 2:**
- mAP von Radar-only mit Distillation (sollte besser sein als WP1 Radar-only)
- Vergleich: Radar-only ohne Distillation vs. mit Distillation
- mAP auf Fog/Rain (sollte etwas besser sein als Stage 1, aber schlechter als Stage 3)

---

### Stage 3: Full Fusion Model Training on Mixed Weather

**Ziel:** Das vollständige Fusions-Modell trainieren mit allen Komponenten:
- LiDAR Encoder (initialisiert mit Teacher-Gewichten)
- Radar Encoder (initialisiert mit distillierten Gewichten)
- Reliability Map
- Spatial Gate
- Shared Backbone + Detection Head

**Konfiguration:**
- Modell: LiDARRadarFusionDetector (mit Gate)
- Daten: Gemischtes Wetter (Clear + Fog + Rain)
- Loss: `L_total = L_detection + lambda_distill * L_distill + lambda_smooth * L_smooth` (optional)
- Teacher: Optional weiterhin frozen oder fein-tuned
- Dauer: ~100–150 Epochen

**Wichtig:** Teacher kann **frozen** bleiben oder optional fein-tuned werden (Ablation).

```python
# Pseudocode für Stage 3
def train_stage3(config):
    # Initialize with Stage 1 + Stage 2 weights
    model = LiDARRadarFusionDetector(config)
    model.lidar_encoder.load_state_dict(torch.load('checkpoint_teacher.pth'))
    model.radar_encoder.load_state_dict(torch.load('checkpoint_radar_distilled.pth'))
    
    # Optional: freeze LiDAR encoder
    if config.freeze_lidar:
        for param in model.lidar_encoder.parameters():
            param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.parameters())
    loader = get_mixed_weather_loader(config)  # Clear + Fog + Rain
    
    for epoch in range(config.epochs):
        for batch in loader:
            output, gate, reliability = model(batch)
            
            # Detection loss
            det_loss = detection_loss(output, batch['labels'])
            
            # Optional: distillation still active
            if config.continue_distill:
                mask = get_doppler_mask(batch['radar_doppler'])
                with torch.no_grad():
                    teacher_features = teacher.encoder(batch['processed_lidar'])
                distill_loss = masked_mse_loss(model.radar_features, teacher_features, mask)
                det_loss += 0.01 * distill_loss  # Reduced weight
            
            # Optional: gate regularization
            loss_total = det_loss
            if config.use_gate_smooth:
                loss_total += config.lambda_smooth * smoothness_loss(gate)
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
    
    # Save final model
    torch.save(model.state_dict(), 'checkpoint_fusion_final.pth')
    return model
```

**Metriken nach Stage 3:**
- mAP auf Clear/Fog/Rain splits
- Vergleich: LiDAR-only vs. Radar-only vs. Naive Fusion vs. Gate without Reliability vs. Full Model
- Gate Heatmaps pro Wetterbedingung
- Ablation: with/without reliability map

---

### 3. Wichtige Konfigurationsparameter

Erstelle eine zentrale Config-Datei für alle Stages:

```yaml
# training_config.yaml

training:
  stage1:
    model: "lidar_only"
    data_split: "clear"
    epochs: 100
    batch_size: 4
    learning_rate: 0.001
    loss: "detection_loss"
  
  stage2:
    model: "radar_only"
    data_split: "clear"
    epochs: 100
    batch_size: 4
    learning_rate: 0.001
    loss: "detection_loss + lambda_distill * masked_distill_loss"
    lambda_distill: 0.1
    teacher_checkpoint: "checkpoint_teacher.pth"
    freeze_teacher: true
  
  stage3:
    model: "lidar_radar_fusion"
    data_split: "mixed"  # Clear + Fog + Rain
    epochs: 150
    batch_size: 4
    learning_rate: 0.001
    freeze_lidar: false  # Optional: can freeze LiDAR encoder
    continue_distill: false  # Optional: reduce distill weight further
    lambda_smooth: 0.01  # Only if gate collapses
    
    # Weight initialization
    init_lidar_from: "checkpoint_teacher.pth"
    init_radar_from: "checkpoint_radar_distilled.pth"
```

---

### 4. Zwischenprüfungen und Validierung

Nach jeder Stage:

#### Nach Stage 1
- [ ] LiDAR-only mAP auf Clear ist **stark** (z.B. > 0.70)
- [ ] Teacher Checkpoint ist gespeichert
- [ ] Teacher ist reproduzierbar (neue Training liefert ähnliche Werte)

#### Nach Stage 2
- [ ] Distilled Radar-only mAP ist **besser** als nicht-distillierter Radar
- [ ] Distillation Loss sinkt über Epochen
- [ ] Radar Checkpoint ist gespeichert
- [ ] Radar zeigt Fortschritt auf Fog/Rain (leicht)

#### Nach Stage 3
- [ ] Fusion mAP ist besser als beste Single-Sensor Baseline
- [ ] Gate-Werte variieren sinnvoll zwischen Wetterbedingungen
- [ ] Keine Gate-Collapse beobachtet (W nicht nur 0 oder 1)
- [ ] Fusion Model Checkpoint ist gespeichert

---

### 5. Erforderliche Baselines und Ablationen

Die Thesis muss folgende Vergleiche durchführen:

#### Baselines (aus WP1)
- [ ] LiDAR-only (Stage 1 output)
- [ ] Radar-only (non-distilled, WP1)
- [ ] Naive Fusion (Concat, WP1)

#### Mit Distillation (aus WP3)
- [ ] Radar-only mit Distillation (Stage 2 output)

#### Mit Gate (aus WP5)
- [ ] Fusion mit Simple Concat (keine Gate)
- [ ] Fusion mit Gate ohne Reliability (Ablation)
- [ ] Fusion mit Gate + Reliability (vollständig, Stage 3)

#### Ablationen für Regularisierung
- [ ] Fusion ohne Gate-Smoothness Regularisierung
- [ ] Fusion mit Gate-Smoothness (nur wenn nötig)

**Erwartete Performance-Ranking:**
```
Fusion (full) > Fusion (no reliability) > Radar (distilled) > Naive Fusion > Radar (no distill)
```

---

### 6. Dokumentationspflichten

1. **Training Configuration Files** – YAML für alle 3 Stages
2. **Training Logs** – Loss-Kurven pro Stage (TensorBoard oder ähnlich)
3. **Stage-wise Results** – Tabelle mit mAP/NDS pro Stage
4. **Final Comparison** – Tabelle aller Baselines und Ablationen
5. **Qualitative Visualizations** – Beispiele von detections unter verschiedenen Wetterbedingungen

---

### 7. Meilenstein für WP6

**Du bist fertig mit WP6, wenn:**

- [ ] Alle drei Stages trainieren ohne Fehler
- [ ] Stage 1 Teacher Checkpoint ist reproduzierbar
- [ ] Stage 2 Radar zeigt Verbesserung durch Distillation
- [ ] Stage 3 Fusion zeigt Verbesserung über alle Wetter-Splits
- [ ] Alle Baselines und Ablationen sind dokumentiert
- [ ] Training Logs und Checkpoints sind archiviert
- [ ] Finale Performance-Tabelle ist erstellt

**Dann** ist die Implementierung des Lite-BEV-Verfahrens abgeschlossen!

---

### 8. Zusammenfassung: Das komplette Training Pipeline

```
WP1 Baselines (Validation)
    ↓
    WP2 Teacher (Stage 1 Training)
    ↓ Checkpoint: teacher_weights
    
    WP3 Distillation (Stage 2 Training)
    ↓ Checkpoint: radar_weights_distilled
    
    WP4 Reliability Map (Inference Module)
    WP5 Gate (Inference Module)
    ↓
    
    WP6 Stage 3: Full Fusion Training
    ↓ Checkpoint: fusion_model_final
    
    Final Evaluation on All Splits
    → Results for Thesis
```

Dies ist der **komplette Trainings-Workflow** des Lite-BEV-Verfahrens.
