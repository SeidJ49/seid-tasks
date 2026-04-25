## Work Package 3: Motion-Aware Radar Distillation – Detaillierte Vorbereitung

Nach WP1 (Baselines) und WP2 (eingefrorener LiDAR Teacher) geht es in WP3 um das **Herzstück der Trainingsphase**:  
Den Radar-Encoder so zu trainieren, dass er **nur in bewegten Regionen** LiDAR-ähnliche geometrische Features lernt – mithilfe einer **Doppler-Maske**.

> **Ziel:** Der Radar-Branch soll später in der Fusion hochwertige BEV-Features liefern, besonders dort, wo LiDAR unter schlechtem Wetter versagt. Aber während des Trainings wird er **nur auf bewegten Objekten** durch den Teacher supervidiert.

---

## 1. Was ist am Ende von WP3 erreicht?

- Ein **distillierter Radar-Encoder** (basierend auf dem Radar‑only Modell aus WP1), der auf Clear‑Weather‑Daten trainiert wurde.
- Die Distillation nutzt eine **Doppler-Maske** (`M_dyn`), die nur Zellen mit bewegten Objekten einschließt.
- Der LiDAR Teacher ist **eingefroren** und liefert die Ziel‑Features.
- Der Radar-Encoder wird **nicht** für die Inference allein verwendet – er wird später in die Fusion eingebaut.

---

## 2. Konkrete technische Schritte (Vorbereitung & Umsetzung)

### Schritt 1: Doppler-Geschwindigkeit aus Radar-Punkten extrahieren

Dein Radar-Rohdatensatz enthält pro Punkt üblicherweise:  
`(x, y, z, v_r)` oder `(x, y, z, v_r, rcs)`.  
`v_r` ist die **radiale Geschwindigkeit** (relativ zum Ego-Fahrzeug).  

**Aufgabe:**  
- Stelle sicher, dass dein Radar-PillarVFE die Doppler-Werte als viertes Feature durchreicht.  
- In der `forward` des Radar-Encoders musst du Zugriff auf die ursprünglichen `v_r` haben, bevor sie im VFE aggregiert werden.  
- **Tipp:** Speichere die Doppler-Werte in `data_dict['radar_doppler']` während der Datenvorverarbeitung.

### Schritt 2: Doppler-Maske `M_dyn` in BEV erzeugen

Du benötigst eine binäre Maske in der gleichen BEV-Auflösung wie die späteren Feature-Maps (`H, W`).

**Algorithmus:**
1. Projiziere alle Radar-Punkte in BEV-Koordinaten (gleiche Pillar-Gitterzellen wie beim Scatter).
2. Für jede Zelle: Wenn mindestens ein Punkt mit `|v_r| > 0.5 m/s` existiert, setze Maske = 1 (dynamisch).  
   (Der Schwellwert ist empirisch – 0.5 m/s filtert stehende Objekte und Rauschen.)
3. Optional: Dilatation der Maske (z.B. 3×3), um auch die unmittelbare Umgebung des bewegten Objekts zu erfassen – das hilft dem Radar, Konturen zu lernen.

**Implementierung in Python (pseudocode):**

```python
def build_doppler_mask(radar_points, bev_shape, point_cloud_range, voxel_size):
    # radar_points: (N, 4) mit x,y,z,v_r
    mask = torch.zeros(bev_shape, dtype=torch.bool)  # (H, W)
    for point in radar_points:
        if abs(point[3]) > 0.5:   # v_r threshold
            # BEV-Koordinaten berechnen (wie beim Pillar Scatter)
            x, y = point[0], point[1]
            # ... mapping zu Gitterzellen
            mask[ix, iy] = 1
    # Optional: dilation
    mask = torch.nn.functional.max_pool2d(mask.float().unsqueeze(0), kernel_size=3, stride=1, padding=1).bool()
    return mask.squeeze(0)  # (H, W)
```

**Wichtig:** Die Maske muss für jedes Batch‑Element separat erzeugt werden.

### Schritt 3: Distillation Loss implementieren

Der Loss wird **nur** auf den Zellen mit `M_dyn == 1` berechnet.

**Einfachste Variante (Masked MSE):**

```python
def masked_mse_loss(radar_bev, teacher_bev, mask):
    # radar_bev, teacher_bev: (B, C, H, W)
    # mask: (B, 1, H, W) binary
    diff = (radar_bev - teacher_bev) ** 2
    masked_diff = diff * mask
    loss = masked_diff.sum() / (mask.sum() + 1e-8)
    return loss
```

**Optional (Masked InfoNCE – kontrastiver Loss):**  
Falls du später mehr herausholen willst, kannst du einen kontrastiven Loss verwenden, der positive Paare (gleiche Zelle, Radar vs. LiDAR) anzieht und negative Paare (verschiedene Zellen) abstößt. Das ist aber nicht zwingend für die erste Version.

**Setze den Distillation-Loss zusätzlich zum ursprünglichen Detektions-Loss des Radar‑only Modells.**  
Gesamtloss = `L_detection + lambda * L_distill`.  
Starte mit `lambda = 0.1` und optimiere später.

### Schritt 4: Trainingspipeline für WP3

- **Trainingsdaten:** Nur Clear‑Weather (weil Teacher nur dafür gut ist).
- **Modell:** Radar‑only PointPillars aus WP1 (Encoder + Backbone + Heads).  
  *Aber:* Der LiDAR Teacher ist eingefroren, und wir nutzen seine BEV-Features als Ziel.
- **Loss:** Radar‑only Detektionsloss + Masked Distillation Loss.
- **Trainingsdauer:** Ähnlich wie WP1 (vielleicht etwas länger, weil zusätzlicher Loss).

**Wichtiger Hinweis:** Der Radar-Encoder wird hier **nicht** in einer Fusion trainiert, sondern eigenständig – nur mit zusätzlicher Supervisionsquelle (Teacher). Das ist das Konzept der Distillation.

### Schritt 5: Evaluierung des distillierten Radar-Modells

- Teste das distillierte Radar‑only Modell auf **Clear** und **Fog/Rain** (obwohl es nur auf Clear trainiert wurde).  
- Erwartung:  
  - Auf Clear: Verbesserung gegenüber dem nicht-distillierten Radar‑only (z.B. +2–5 mAP).  
  - Auf Fog/Rain: Leichte Verbesserung (da der Teacher klare Geometrie eingeprägt hat).  
- Visualisiere die BEV-Features vor und nach der Distillation (z.B. Heatmaps einer bestimmten Kanalebene). Zeige, dass der distillierte Radar jetzt „LiDAR-ähnliche“ Konturen von bewegten Autos hat.

---

## 3. Was ist *nicht* Teil von WP3?

- Keine Fusion mit LiDAR (das kommt in WP5).
- Keine Reliability Map (WP4).
- Kein Gate.
- Kein Training auf gemischten Wetterdaten (nur Clear).
- Keine Änderung am LiDAR Teacher (der bleibt frozen).

---

## 4. Dokumentationspflichten (laut Thesis)

- **Doppler-Masken-Modul** – Code und Beschreibung, wie die Maske erzeugt wird.
- **Custom Distillation Loss** – Implementierung (z.B. `MaskedMSELoss`).
- **Trainings-Logs** – Zeigen, dass der Distillation Loss sinkt und die Detektion sich verbessert.
- **Qualitative Visualisierungen** – Vorher/Nachher der Radar BEV Features (z.B. für eine Szene mit einem fahrenden Auto).  
  - Vor Distillation: Radar zeigt nur verrauschte Punkte.  
  - Nach Distillation: Radar zeigt eine klare, gefüllte Kontur ähnlich wie LiDAR.

---

## 5. Typische Fallstricke und Lösungen

| Problem | Lösung |
|---------|--------|
| Radar hat nur sehr wenige bewegte Punkte (spärliche Maske) | Verwende eine größere Dilatation (z.B. 5×5) oder einen weichen Schwellwert (Gewichtung nach |v_r|). |
| Distillation verschlechtert die Radar‑Detektion auf Clear | Reduziere `lambda` (Gewicht des Distillation Loss) oder friere frühe Layer des Radar-Encoders ein. |
| Teacher‑Features haben andere Dimension als Radar‑Features | Du kannst eine 1×1 Conv auf die Teacher-Features projizieren, um die Kanalzahl anzupassen. |
| Doppler-Werte sind nicht kalibriert (viele Ausreißer) | Verwende einen Median-Filter über die v_r pro Zelle oder setze einen höheren Schwellwert (z.B. 1.0 m/s). |
| Keine Verbesserung unter Fog/Rain | Das ist okay – die Distillation findet nur auf Clear statt. Die Robustheit kommt später durch das Gate in WP5. |

---

## 6. Meilenstein für WP3

**Du bist fertig mit WP3, wenn:**

- [ ] Du eine Funktion `build_doppler_mask` hast, die eine binäre BEV-Maske für bewegte Objekte erzeugt.
- [ ] Der Distillation Loss (`MaskedMSE`) ist implementiert und in das Training des Radar‑only Modells integriert.
- [ ] Du hast ein **distilliertes Radar‑only Modell** trainiert (Checkpoint gespeichert).
- [ ] Du kannst zeigen (z.B. mit einer Plot-Script), dass die Radar-BEV-Features nach Distillation in den maskierten Regionen den LiDAR-Features ähneln.
- [ ] Du hast eine Tabelle mit mAP des distillierten vs. nicht-distillierten Radar‑only auf Clear und Fog/Rain.

**Dann** bist du bereit für **WP4 (LiDAR Reliability Estimation)** und **WP5 (Reliability-Guided Gated Fusion)**.

---

## 7. Ausblick: Wie WP3 in WP5 eingebaut wird

Später in WP5 initialisierst du den Radar-Zweig der Fusion **mit den Gewichten des distillierten Radar-Encoders**.  
Der LiDAR-Zweig wird mit den Gewichten des LiDAR Teachers initialisiert (oder einem frisch trainierten LiDAR‑only auf gemischten Daten).  
Dann trainierst du die gesamte Fusion mit dem Gate – aber der Radar-Encoder startet bereits mit einem guten, LiDAR-ähnlichen Feature-Raum.

Ohne WP3 wäre der Radar-Zweig zu schwach, um das Gate sinnvoll zu nutzen.