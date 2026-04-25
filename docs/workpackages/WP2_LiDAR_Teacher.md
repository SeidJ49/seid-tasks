## Work Package 2: LiDAR Teacher Model – Detaillierte Beschreibung

Nachdem WP1 die reinen Baselines (LiDAR‑only, Radar‑only, naive Fusion) liefert, geht es in WP2 um die Vorbereitung des **LiDAR Teachers**. Dieser Teacher wird in WP3 genutzt, um die Radar‑Features durch Distillation zu verbessern.

---

### 1. Ziel von WP2

Ein **starkes, eingefrorenes LiDAR‑Modell** bereitstellen, das auf **Clear‑Weather‑Daten** trainiert ist und später als „Ground Truth“‑Feature‑Extraktor für die Radar‑Distillation dient.

Der Teacher wird **nur** für die Distillation benötigt – nicht für die finale Inferenz.

---

### 2. Konkrete technische Schritte

#### Schritt 1: LiDAR‑only Modell aus WP1 nehmen

Du verwendest das **bereits trainierte LiDAR‑only PointPillars** aus WP1 – aber mit einer wichtigen Änderung:  
Es muss auf **ausschließlich Clear‑Weather‑Daten** trainiert sein (nicht auf gemischten Wetterdaten). Falls dein Datensatz klare Wetter‑Labels hat, filtere die Trainings‑Samples entsprechend.

**Warum?**  
Der Teacher soll ideale Geometrie in gutem Wetter lernen. Wenn er schon Regen/Fog sieht, könnte er degradierte Features lernen, die dann die Distillation verschlechtern.

#### Schritt 2: Teacher einfrieren

Nach dem Training setzt du alle Parameter des LiDAR‑Modells auf `requires_grad = False`.  
Der Teacher wird während der Distillation (WP3) **nicht** mehr trainiert.

In PyTorch:

```python
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False
```

#### Schritt 3: Feature‑Ebene für die Distillation auswählen

Die Thesis sagt: „Select the **BEV feature map** that will be used as teacher target.“  
Du musst dich für eine bestimmte Aktivierung im Teacher entscheiden. Typische Kandidaten:

| Ebene | Beschreibung | Vor‑/Nachteile |
|-------|--------------|----------------|
| **Nach PointPillarScatter** (rohe BEV‑Features) | `spatial_features` direkt nach Scatter | + Hohe räumliche Auflösung<br>- Viele Kanäle (z.B. 64) |
| **Nach Backbone (aber vor Heads)** | `spatial_features_2d` | + Stärker abstrahiert<br>- Geringere Auflösung (durch Downsampling) |

**Empfehlung aus der Thesis:** Starte mit den **BEV‑Features nach dem Scatter** (also vor dem Backbone).  
Diese sind räumlich hoch aufgelöst und enthalten noch die Pillar‑Struktur – ideal für die Distillation auf BEV‑Ebene.

Dokumentiere:

- Welcher Layer genau (Variable‑Name in deinem Code)
- Shape: `(B, C, H, W)` – notiere `C`, `H`, `W`
- Wie greifst du im Forward darauf zu?

#### Schritt 4: Teacher‑Checkpoint speichern

Speichere den gesamten Teacher‑Zustand (weights + config) unter einem eindeutigen Namen, z.B.:

```
checkpoints/lidar_teacher_clear_weather.pth
```

Zusätzlich speichere die zugehörige Config (`lidar_only_pointpillars.yaml`).

#### Schritt 5: Validierung des Teachers

Überprüfe, dass der Teacher auf **Clear‑Weather‑Val** gute Werte liefert (erwarte: ähnlich wie WP1 LiDAR‑only auf Clear).  
Auf **Fog/Rain** darf er schlechter sein – das ist gewünscht, zeigt aber die Degradation.

Erstelle eine kleine Tabelle:

| Wetterbedingung | mAP (Teacher) |
|----------------|---------------|
| Clear          | 0.72          |
| Fog            | 0.34          |
| Rain           | 0.41          |

---

### 3. Was ist *nicht* Teil von WP2?

- **Keine** Distillation (das ist WP3)
- **Kein** Radar‑Training
- **Keine** Fusion, kein Gate, keine Reliability
- **Keine** Änderung an der Radar‑Architektur

WP2 ist reine Vorbereitung: ein starkes, eingefrorenes LiDAR‑Modell bereitstellen, das in WP3 als „Lehrer“ fungiert.

---

### 4. Dokumentationspflichten (laut Thesis)

Du musst folgendes abgeben:

1. **Trained teacher checkpoint** (`.pth`-Datei)
2. **Config‑Datei** (die zum Training verwendet wurde)
3. **Feature‑Level‑Dokumentation**:  
   - Welcher Layer wurde als Teacher‑Target gewählt?  
   - Begründung (z.B. „hohe räumliche Auflösung für feine geometrische Strukturen“)
   - Code‑Snippet, das zeigt, wie du diese Features im Forward extrahierst.
4. **Validierungsperformance** (Tabelle mit mAP pro Wetterbedingung)

---

### 5. Typische Fallstricke und Lösungen

| Problem | Lösung |
|---------|--------|
| Clear‑Weather‑Daten sind nicht explizit gelabelt | Nutze Metadaten (z.B. Wetter‑Strings aus nuScenes: `'sunny'`, `'clear'`). Falls nicht vorhanden, trainiere auf allen Daten – aber dokumentiere die Einschränkung. |
| Teacher hat gleiche Architektur wie Student (Radar‑Branch) | Das ist gewünscht: Der Radar‑Branch soll später dieselbe Struktur haben, aber andere Gewichte. |
| Teacher‑Features sind zu groß (Speicher) | Du kannst die Features vor der Distillation mit einer 1×1‑Conv auf weniger Kanäle projizieren – aber das ist optional und erst in WP3 relevant. |
| Teacher verliert an Performance, weil er nur auf Clear trainiert ist | Das ist beabsichtigt – er soll ideale Geometrie lernen, nicht wetter‑robust sein. |

---

### 6. Meilenstein für WP2

**Du bist fertig mit WP2, wenn:**

- [ ] Ein LiDAR‑only Modell **nur auf Clear‑Weather** trainiert wurde.
- [ ] Das Modell ist eingefroren (`eval()`, `requires_grad=False`).
- [ ] Du hast eine klare Dokumentation, welcher Layer als Teacher‑Target dient (Shape, Zugriff im Code).
- [ ] Der Teacher‑Checkpoint ist gespeichert und kann von einem separaten Skript geladen werden (Test: `model.load_state_dict(torch.load(...))`).
- [ ] Die Validierungs‑Metriken auf Clear, Fog, Rain liegen vor.

**Dann** kannst du zu WP3 (Motion‑Aware Radar Distillation) übergehen.

---

### 7. Verbindung zu WP3 (kurzer Ausblick)

In WP3 wirst du:

- Den Radar‑Encoder (aus WP1 Radar‑only) nehmen.
- Den **eingefrorenen Teacher** verwenden, um dessen BEV‑Features als Ziel zu nutzen.
- Eine Doppler‑basierte Maske erzeugen, um die Distillation **nur auf bewegten Objekten** anzuwenden.
- Einen Masked Loss (`Masked_MSE` oder `Masked_InfoNCE`) implementieren.

Ohne WP2 kannst du WP3 nicht starten – deshalb ist dieser Schritt essenziell.

---

### 8. Beispiel‑Code für die Teacher‑Feature‑Extraktion

```python
class LidarTeacher(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pillar_vfe = PillarVFE(...)
        self.scatter = PointPillarScatter(...)
        self.backbone = BaseBEVBackbone(...)  # optional, falls du tiefere Features nutzt
        
        # Nach dem Training: einfrieren
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, data_dict):
        # Nur LiDAR-Zweig
        batch_dict = self.pillar_vfe(data_dict['processed_lidar'])
        batch_dict = self.scatter(batch_dict)
        teacher_bev = batch_dict['spatial_features']  # <- das ist das Target
        return teacher_bev
```

In WP3 wirst du dann `teacher_bev` als Ziel für die Radar‑Features verwenden.