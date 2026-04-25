## Work Package 4: LiDAR Reliability Estimation – Detaillierte Beschreibung

Nachdem WP3 den Radar-Encoder durch Distillation verbessert hat, geht es in WP4 um die Erkennung, wo die LiDAR-Features unter schlechtem Wetter unzuverlässig werden.

---

### 1. Ziel von WP4

Eine **Zuverlässigkeits-Karte** (`U_L`) erstellen, die räumlich anzeigt, wo LiDAR-BEV-Features durch Regen, Nebel oder andere Wetter beeinträchtigt werden.

Diese Karte wird später in WP5 als Eingabe für den Gate-Mechanismus verwendet.

---

### 2. Wichtige Erwartung: Nicht nur lokale Varianz

### Problem mit reiner Feature-Varianz

Die reine räumliche Varianz der LiDAR-Features kann auch folgende Effekte erfassen:
- Objektgrenzen
- Clutter/Rauschen
- Starke geometrische Strukturen
- Hochkontrast-Kanten

Dies ist **kein** zuverlässiger Indikator für Wetter-Degradation allein.

**Daher: Verwende physikalisch interpretierbare Heuristiken!**

---

### 3. Konkrete technische Schritte

#### Schritt 1: Kandidaten für die Reliabilität

Die erste Implementierung sollte eine oder mehrere dieser physikalischen Metriken verwenden:

| Metrik | Beschreibung | Vorteile | Nachteile |
|--------|-------------|----------|-----------|
| **Point Count** | Anzahl LiDAR-Punkte pro Pillar | + Einfach zu berechnen<br>+ Direkt mit Visibility korreliert | - Kann auch starke Objekte anzeigen |
| **Occupancy Sparsity** | Anteil leerer Pillars in Nachbarschaft | + Schlechtes Wetter → mehr leere Zellen | - Auch von Objektverteilung abhängig |
| **Local Density Drop** | Varianz der Punktdichte in 3×3-Fenster | + Nebel/Regen erzeugt flache Dichte | - Komplex zu implementieren |
| **Mean Intensity** | Durchschnittliche Intensität der Punkte | + Regen/Nebel reduziert Intensity | - Dataset-abhängig |
| **Neighborhood Consistency** | Konsistenz mit Nachbarpillars | + Wetter-Degradation erzeugt Inkonsistenz | - Schwer zu quantifizieren |

**Empfehlung:** Kombiniere **Point Count + Occupancy Sparsity** als erste Version.

#### Schritt 2: Reliability-Map-Formel (erste Version)

```
U_L[i,j] = 1 - f(point_count[i,j], occupancy[i,j])
```

wobei:
- `point_count[i,j]` = Anzahl LiDAR-Punkte in Pillar (i,j)
- `occupancy[i,j]` = Anteil besetzter Pillars in 3×3-Nachbarschaft um (i,j)
- `f()` = Normalisierungsfunktion, die beide auf [0,1] skaliert

**Interpretierung:**
- **Hoher U_L Wert** = niedrige LiDAR-Zuverlässigkeit (schlechtes Wetter vermutet)
- **Niedriger U_L Wert** = hohe LiDAR-Zuverlässigkeit (gutes Wetter)

Beispiel:
```python
def compute_reliability_map(lidar_points, bev_shape, point_cloud_range, voxel_size):
    # Berechne Point Count pro Pillar
    point_count = torch.zeros(bev_shape)
    for point in lidar_points:
        # Pillar-Koordinaten berechnen
        ix, iy = project_to_bev(point, point_cloud_range, voxel_size)
        point_count[ix, iy] += 1
    
    # Berechne Occupancy (Anteil besetzter Pillars in Nachbarschaft)
    occupancy = (point_count > 0).float()
    occupancy_blur = torch.nn.functional.max_pool2d(
        occupancy.unsqueeze(0), kernel_size=3, stride=1, padding=1
    ).squeeze(0)
    
    # Kombiniere: U_L = 1 - (normalized_point_count * occupancy_blur)
    normalized_pc = torch.tanh(point_count / max_expected_points)
    U_L = 1 - (normalized_pc * occupancy_blur)
    return U_L  # Shape: (H, W)
```

#### Schritt 3: Visualisierung und Validierung

Erstelle Visualisierungen der Reliability-Map unter verschiedenen Wetterbedingungen:

1. **Clear Weather:** U_L sollte meist **niedrig** sein (zuverlässig)
2. **Light Fog:** U_L sollte **moderat** sein
3. **Heavy Rain:** U_L sollte **hoch** sein (wenig zuverlässig)

Erstelle eine Heatmap-Tabelle:

| Wetterbedingung | Durchschnittlicher U_L | Min U_L | Max U_L |
|----------------|------------------------|---------|---------|
| Clear          | 0.15                   | 0.01    | 0.45    |
| Light Fog      | 0.35                   | 0.05    | 0.75    |
| Heavy Rain     | 0.62                   | 0.20    | 0.95    |

#### Schritt 4: Korrelation mit Degradation

Zeige, dass U_L tatsächlich mit Wetter-Degradation korreliert:

```python
# Berechne Spearman Korrelation zwischen U_L und Detection Performance (mAP)
correlation = spearmanr(mean_U_L_per_scene, mAP_per_scene)
print(f"Korrelation(U_L, mAP_Degradation): {correlation}")
```

Erwartung: **Negative Korrelation** (höher U_L → niedriger mAP unter schlechtem Wetter).

---

### 4. Was ist *nicht* Teil von WP4?

- **Keine** gelernte Reliability Head (das ist optional, späters)
- **Keine** Supervision durch externe Qualitätslabels
- **Keine** Feature-Varianz als Hauptsignal
- **Keine** Integration in den Gate-Mechanismus (das ist WP5)

WP4 ist reine Erstellung einer heuristischen Reliability-Map.

---

### 5. Dokumentationspflichten

1. **Reliability-Map-Modul** – Code mit ausführlicher Dokumentation
2. **Heuristische Auswahl** – Begründung, warum diese Metriken gewählt wurden
3. **Visualisierungen** – Heatmaps für Clear/Fog/Rain
4. **Korrelationsanalyse** – Beweis, dass U_L mit Wetter-Degradation korreliert

---

### 6. Meilenstein für WP4

**Du bist fertig mit WP4, wenn:**

- [ ] Eine Funktion `compute_reliability_map()` implementiert ist
- [ ] Die Reliability-Map räumlich variiert zwischen Clear und schlechtem Wetter
- [ ] Du hast Heatmap-Visualisierungen für mindestens 3 Wetterbedingungen
- [ ] Du hast eine Korrelationsanalyse (U_L vs. mAP) durchgeführt
- [ ] Die Reliability-Map ist in das BEV-Pipeline des Detektors integriert

**Dann** bist du bereit für **WP5 (Reliability-Guided Gated Fusion)**.
