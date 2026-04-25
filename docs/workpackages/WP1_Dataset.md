## Work Package 1: Baseline Setup – Detaillierte Beschreibung

Ziel: **Stabile, reproduzierbare Baselines ohne jegliche Fusion oder Distillation** aufbauen. Erst wenn diese laufen, wird die eigene Methode implementiert.

---

### 1. Was muss am Ende funktionieren?

Drei unabhängige Detektoren, alle auf **PointPillars**-Basis:

| Modell | Eingang | Ausgabe |
|--------|---------|---------|
| **LiDAR‑only** | Nur LiDAR-Punkte | 3D‑Bounding‑Boxes |
| **Radar‑only** | Nur Radar-Punkte | 3D‑Bounding‑Boxes |
| **Naive Fusion** | LiDAR + Radar (kein Gate, keine Reliability) | 3D‑Bounding‑Boxes |

Alle drei müssen auf **demselben Dataset** (z.B. nuScenes, View-of-Delft, oder eurem internen Datensatz) trainierbar sein, mit klaren Metriken (mAP, NDS, etc.).

---

### 2. Konkrete technische Schritte

#### Schritt 1: Code-Struktur vorbereiten

Erstelle in deinem OpenPCDet‑artigen Framework drei Konfigurationsdateien:

- `cfgs/lidar_only_pointpillars.yaml`
- `cfgs/radar_only_pointpillars.yaml`
- `cfgs/naive_fusion_pointpillars.yaml`

Lege für jedes Modell eine eigene Klasse an (z.B. in `models/`):

```python
class LidarOnlyPointPillars(nn.Module):
    # Nur LiDAR-PillarVFE → Scatter → Backbone → Heads

class RadarOnlyPointPillars(nn.Module):
    # Gleiche Architektur, aber Radar-Eingang

class NaiveFusionPointPillars(nn.Module):
    # Beide PillarVFE + Scatter, dann Concat der BEV-Features (kein Gate!)
    # Danach gemeinsamer Backbone + Heads
```

#### Schritt 2: Daten-Loader anpassen

Dein `data_dict` muss enthalten:

- Für LiDAR‑only: nur `processed_lidar` (voxel_features, coords, num_points)
- Für Radar‑only: nur `processed_radar`
- Für naive Fusion: beide, aber ohne jegliche Fusion außer **channel‑weises Concat** der BEV-Features.

**Wichtig:** Für Radar‑only musst du Radar-Punkte genau wie LiDAR-Punkte in Pillars voxeln. Radar hat oft 4 Merkmale (x, y, z, Doppler oder RCS). Stelle sicher, dass `num_point_features` in `pillar_vfe` passend gesetzt ist.

#### Schritt 3: BEV-Feature-Extraktion identifizieren und dokumentieren

Du musst genau wissen:

- Nach dem `PointPillarScatter`: Wie groß ist der BEV‑Tensor?  
  Format: `(B, C, H, W)` – notiere `C`, `H`, `W`.
- Welche räumliche Auflösung hat eine Zelle? (z.B. 0.2 m pro Pixel)
- Wo im Code greifst du später auf diese `spatial_features` zu?

Dokumentiere das in einer Textdatei oder Markdown-Tabelle.

#### Schritt 4: Training durchführen

- **LiDAR‑only** auf **Clear‑Weather**-Daten trainieren (falls vorhanden, sonst gemischt).  
  Mindestens so lange, bis die Loss-Kurven konvergieren.
- **Radar‑only** auf **gleichen Daten** trainieren – gleiche Epochen, gleiche Augmentations.
- **Naive Fusion** ebenfalls trainieren.

**Erwartung:** Radar‑only wird deutlich schlechter sein als LiDAR‑only. Das ist normal und erwünscht – zeigt den Bedarf an Distillation.

#### Schritt 5: Evaluieren und Ergebnisse festhalten

Führe jedes Modell auf **mindestens** den Weather‑Splits aus:

- Clear
- Fog (light/heavy, falls verfügbar)
- Rain (light/heavy)

Notiere für jeden:

- mAP (oder eure Hauptmetrik)
- ggf. Inferenzzeit (FPS) – das ist später wichtig für den „lightweight“-Anspruch

Erstelle eine **Ergebnistabelle** (z.B. in Excel oder Markdown):

| Modell | Clear mAP | Fog mAP | Rain mAP | FPS |
|--------|-----------|---------|----------|-----|
| LiDAR‑only | 0.72 | 0.34 | 0.41 | 25 |
| Radar‑only | 0.38 | 0.36 | 0.39 | 24 |
| Naive Fusion | 0.70 | 0.45 | 0.50 | 22 |

---

### 3. Was ist *nicht* Teil von WP1?

- **Kein** Gate, keine Reliability-Map, kein Gating.
- **Keine** Distillation (auch keine Doppler-Maske).
- **Kein** eingefrorener Teacher.
- **Kein** gestuftes Training.
- **Keine** speziellen Loss-Funktionen.

Alles, was über einfaches Concat der BEV-Features hinausgeht, gehört zu späteren WPs.

---

### 4. Dokumentationspflichten (laut Thesis)

Du musst folgendes abgeben:

1. **Config‑Dateien** (vollständig, mit allen Hyperparametern)
2. **Trainings‑Logs** (z.B. TensorBoard‑Events oder Text‑Logs mit Loss, mAP pro Epoche)
3. **Ergebnistabelle** (s.o.)
4. **Dokumentation der BEV‑Tensor‑Shapes** – wo genau im Code die Features vor dem Backbone liegen, z.B.:

   ```python
   # In NaiveFusionPointPillars.forward():
   lidar_bev = self.lidar_scatter(lidar_batch_dict)['spatial_features']  # shape: (B, 64, 200, 176)
   radar_bev = self.radar_scatter(radar_batch_dict)['spatial_features']  # shape: (B, 64, 200, 176)
   fused_bev = torch.cat([lidar_bev, radar_bev], dim=1)  # shape: (B, 128, 200, 176)
   # fused_bev geht dann in self.backbone
   ```

   Halte diese Shapes fest – sie sind Grundlage für WP4 (Reliability) und WP5 (Gate).

5. **Kurze Analyse**: Warum ist Radar‑only so viel schlechter? (weniger Punkte, kein Doppler genutzt, Rauschen, …)

---

### 5. Typische Fallstricke und Lösungen

| Problem | Lösung |
|---------|--------|
| Radar‑Punkte haben nur 3 Koordinaten + Doppler | Passe `num_point_features` in `PillarVFE` auf 4 (oder 5 falls RCS). Fehlende Werte mit 0 auffüllen. |
| Radar‑PillarVFE produziert NaNs wegen zu weniger Punkte pro Pillar | Erhöhe `max_num_points_per_pillar` oder setze minimale Punktzahl auf 1. |
| Naive Fusion ist nicht besser als LiDAR‑only | Das ist okay – zeigt, dass einfaches Concat nicht ausreicht. Notiere es als Baseline. |
| Unterschiedliche BEV-Größen durch unterschiedliche Point‑Ranges | Stelle sicher, dass `point_cloud_range` für LiDAR und Radar identisch ist (z.B. [-50, -50, -3, 50, 50, 1]). |
| Training bricht ab wegen Speichermangel | Reduziere Batch‑Size, oder verwende Gradient Accumulation. |

---

### 6. Meilenstein für WP1

**Du bist fertig mit WP1, wenn:**

- [ ] Alle drei Modelle laufen ohne Fehler durch eine komplette Epoche.
- [ ] Die Ergebnisse (mAP) sind reproduzierbar – zweimaliges Training liefert ähnliche Werte (±1%).
- [ ] Du die BEV‑Tensor‑Shapes dokumentiert hast.
- [ ] Du eine Tabelle mit Ergebnissen unter Clear, Fog, Rain vorliegen hast.

**Dann** gehst du zu WP2 (LiDAR Teacher Modell).

---

### 7. Beispiel für einen Minimal‑Check (Selbsttest)

Führe folgenden Test in einer Python‑Shell aus (nach dem Training):

```python
# Lade dein LiDAR‑only Modell
model_lidar = LidarOnlyPointPillars(cfg)
data = next(iter(val_loader))
out = model_lidar(data)
print(out['cls_preds'].shape)   # sollte (B, anchor_number, H, W) sein
print(out['reg_preds'].shape)   # (B, 7*anchor_number, H, W)

# Gleiche für Radar‑only und naive Fusion
```

Wenn das klappt, ist WP1 technisch abgeschlossen.