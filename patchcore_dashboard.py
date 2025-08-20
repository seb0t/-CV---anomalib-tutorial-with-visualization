#!/usr/bin/env python3
"""
🔬 PatchCore Educational Dashboard
==================================

Dashboard web interattiva per l'analisi delle anomalie usando PatchCore.
Parte della piattaforma educativa per la comprensione pratica degli 
algoritmi di anomaly detection.

Features:
- ✅ Heatmap interattive con hover dettagliato
- ✅ Distribuzione scores in tempo reale  
- ✅ Controlli slider responsive
- ✅ Immagini cliccabili per test case selection
- ✅ Colormap con scala fissa per confronti
- ✅ Tema scuro professionale

Requisiti:
- dash >= 3.2.0
- plotly >= 5.17.0
- numpy >= 1.24.0
- Pillow >= 10.0.0

Uso:
    python patchcore_dashboard.py

La dashboard sarà disponibile su: http://localhost:8051
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def create_sample_data():
    """Crea dati di esempio per la dashboard con dataset esteso"""
    print("📦 Creazione dati di esempio...")
    np.random.seed(42)
    
    def draw_smiley(size=20, eye_offset=0, mouth_curve=0.5, eye_size=1, rotation=0, noise_std=0.08):
        """Genera uno smile con variazioni parametriche per diversità"""
        img = np.ones((size, size), dtype=np.float32)
        
        # Calcola posizioni degli occhi con offset (con bounds checking)
        left_eye_x = max(0, min(5 + eye_offset, size - 2))
        right_eye_x = max(0, min(size - 7 + eye_offset, size - 2))
        eye_y = 5
        
        # Disegna occhi con dimensione variabile (con bounds checking migliorato)
        eye_end = min(eye_y + 1 + eye_size, size)
        
        # Occhio sinistro
        left_eye_end = min(left_eye_x + 1 + eye_size, size)
        if left_eye_x >= 0 and left_eye_end <= size and eye_y >= 0 and eye_end <= size:
            img[eye_y:eye_end, left_eye_x:left_eye_end] = 0
        
        # Occhio destro (correzione del bug)
        right_eye_end = min(right_eye_x + 1 + eye_size, size)
        if right_eye_x >= 0 and right_eye_end <= size and eye_y >= 0 and eye_end <= size:
            img[eye_y:eye_end, right_eye_x:right_eye_end] = 0
        
        # Bocca curva con parametro di curvatura
        mouth_start = 6
        mouth_end = size - 6
        for x in range(mouth_start, mouth_end):
            y_offset = mouth_curve * np.sin((x - mouth_start) / (mouth_end - mouth_start) * np.pi) * 4
            y = int((size - 6) + y_offset)
            if 0 <= y < size:
                img[y, x] = 0
        
        # Rotazione semplice se richiesta
        if rotation != 0:
            # Piccola rotazione simulata spostando pattern
            shift = int(rotation)
            if shift > 0:
                img = np.roll(img, shift, axis=1)
            elif shift < 0:
                img = np.roll(img, shift, axis=0)
        
        # Aggiungi rumore
        noise = np.random.normal(0, noise_std, (size, size)).astype(np.float32)
        img = np.clip(img + noise, 0, 1)
        return img
    
    def to_rgb(img):
        return np.stack([img]*3, axis=0)
    
    print("🖼️ Generazione 20 immagini 'good' con variazioni...")
    # Genera 20 immagini "good" con variazioni
    good_images = []
    for i in range(20):
        # Varia parametri per creare diversità
        eye_offset = np.random.randint(-1, 2)
        mouth_curve = np.random.uniform(0.3, 0.8)
        eye_size = np.random.randint(0, 2)
        rotation = np.random.randint(-1, 2)
        noise_level = np.random.uniform(0.05, 0.12)
        
        img = draw_smiley(20, eye_offset, mouth_curve, eye_size, rotation, noise_level)
        good_images.append(to_rgb(img))
    
    print("🚨 Generazione 5 immagini anomale...")
    # Genera 5 immagini con anomalie evidenti
    anomaly_images = []
    for i in range(5):
        img = draw_smiley(20, 0, 0.5, 1, 0, 0.08)
        
        # Aggiungi diverse tipologie di anomalie
        if i == 0:
            img[3:5, 16:18] = 0  # Macchia nera
        elif i == 1:
            img[8:12, 8:12] = 1  # Macchia bianca
        elif i == 2:
            img[15:17, 3:8] = 0  # Linea anomala
        elif i == 3:
            img[2:4, 2:4] = 0.5  # Pixel grigi
        else:
            img[10:15, 15:18] = np.random.random((5, 3))  # Rumore localizzato
        
        anomaly_images.append(to_rgb(img))
    
    print("✅ Aggiunta 5 immagini 'good' per confronto...")
    # Aggiungi altre 5 immagini good per il confronto
    additional_good = []
    for i in range(5):
        eye_offset = np.random.randint(-1, 2)
        mouth_curve = np.random.uniform(0.4, 0.7)
        eye_size = np.random.randint(0, 2)
        rotation = np.random.randint(-1, 2)
        noise_level = np.random.uniform(0.06, 0.10)
        
        img = draw_smiley(20, eye_offset, mouth_curve, eye_size, rotation, noise_level)
        additional_good.append(to_rgb(img))
    
    # Mescola le immagini per test (5 anomalie + 5 good)
    test_images = anomaly_images + additional_good
    test_labels = ['Anomaly'] * 5 + ['Good'] * 5
    
    # Mescola insieme
    indices = list(range(10))
    np.random.shuffle(indices)
    test_images_mixed = [test_images[i] for i in indices]
    test_labels_mixed = [test_labels[i] for i in indices]
    
    print("✅ Dati di esempio creati!")
    print(f"   📋 20 immagini 'good' per training")
    print(f"   ⚠️ 5 immagini anomale per test")
    print(f"   ✅ 5 immagini 'good' per confronto")
    print(f"   🎯 Totale immagini selezionabili: 10 (mescolate)")
    
    return good_images, test_images_mixed, test_labels_mixed

# Dati globali per la dashboard
good_images, test_images_mixed, test_labels_mixed = create_sample_data()
current_test_index = 0  # Indice dell'immagine di test corrente

# Scala fissa per la colormap delle anomalie
ANOMALY_SCALE_MIN = 0.0
ANOMALY_SCALE_MAX = 8.0  # Massimo empirico per patch 20x20 con valori 0-1

def calculate_anomaly_map(patch_size, stride, test_image):
    """Calcola la mappa delle anomalie ottimizzata con scala fissa"""
    anomaly_map = np.zeros((20, 20))
    scores = []
    patches_analyzed = 0
    
    for i in range(0, 20 - patch_size + 1, stride):
        for j in range(0, 20 - patch_size + 1, stride):
            # Patch di test
            patch_test = test_image[:, i:i+patch_size, j:j+patch_size].flatten()
            
            # Patch normali (usa le 20 immagini good)
            patches_norm = [img[:, i:i+patch_size, j:j+patch_size].flatten() 
                          for img in good_images]
            
            # Calcola distanze euclidee
            dists = [np.linalg.norm(patch_test - p) for p in patches_norm]
            score = np.min(dists)
            
            # Clamp del score per mantenere scala fissa
            score = np.clip(score, ANOMALY_SCALE_MIN, ANOMALY_SCALE_MAX)
            
            # Assegna score alla mappa
            anomaly_map[i:i+patch_size, j:j+patch_size] = score
            scores.append(score)
            patches_analyzed += 1
    
    return anomaly_map, scores, patches_analyzed

# Creazione app Dash con API aggiornata
app = dash.Dash(__name__)
app.title = "🔬 PatchCore Dashboard"

# Layout della dashboard con styling migliorato
app.layout = html.Div([
    # Header compatto
    html.Div([
        html.H1("🔬 PatchCore Dashboard Interattiva", 
                style={'textAlign': 'center', 'color': '#00d4ff', 'marginBottom': 5, 'fontSize': '18px'}),
        html.P("Analisi real-time delle anomalie con dataset esteso (20 Good + 10 Test)", 
               style={'textAlign': 'center', 'color': '#888', 'marginBottom': 8, 'fontSize': '12px'})
    ], style={'backgroundColor': '#2a2a2a', 'padding': '6px', 'borderRadius': '6px', 'marginBottom': '6px'}),
    
    # Sezione 20 immagini good
    html.Div([
        html.H3("📋 Training Dataset (20 Good Images)", style={'color': '#4ecdc4', 'fontSize': '14px', 'marginBottom': '8px'}),
        html.Div(id='good-images-display', style={
            'display': 'grid', 
            'gridTemplateColumns': 'repeat(20, 1fr)', 
            'gap': '1px',
            'marginBottom': '10px'
        })
    ], style={'backgroundColor': '#2a2a2a', 'padding': '8px', 'borderRadius': '6px', 'marginBottom': '6px'}),
    
    # Sezione immagini test cliccabili (senza dropdown)
    html.Div([
        html.H3("🎯 Test Images (5 Anomaly + 5 Good - Click to Select)", style={'color': '#ff6b6b', 'fontSize': '14px', 'marginBottom': '8px'}),
        html.Div(id='test-images-display', style={
            'display': 'grid', 
            'gridTemplateColumns': 'repeat(10, 1fr)', 
            'gap': '3px',
            'marginBottom': '10px'
        }),
        # Store per l'immagine selezionata
        dcc.Store(id='selected-test-image', data=0)
    ], style={'backgroundColor': '#2a2a2a', 'padding': '8px', 'borderRadius': '6px', 'marginBottom': '6px'}),
    
    # Grafici principali (ridotti ulteriormente)
    dcc.Graph(id='dashboard-plots', style={'height': '320px', 'marginBottom': '8px'}),
    
    # Sezione inferiore: Metriche a sinistra (compatte), Controlli a destra
    html.Div([
        # Metriche real-time (sinistra - più piccole)
        html.Div(id='metrics-display', style={
            'flex': '0 0 30%', 'marginRight': '3%',
            'backgroundColor': '#2a2a2a', 'padding': '8px', 'borderRadius': '6px',
            'color': '#e0e0e0'
        }),
        
        # Controlli slider (destra - più grandi)
        html.Div([
            html.H4("🎛️ Controlli", style={'color': '#00d4ff', 'marginBottom': 8, 'fontSize': '14px'}),
            html.Div([
                html.Label("📏 Patch Size:", style={'color': '#e0e0e0', 'fontWeight': 'bold', 'marginBottom': 3, 'fontSize': '11px'}),
                dcc.Slider(
                    id='patch-size-slider',
                    min=2, max=10, step=1, value=5,
                    marks={i: {'label': str(i), 'style': {'color': '#e0e0e0', 'fontSize': '9px'}} for i in range(2, 11)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'marginBottom': 8}),
            html.Div([
                html.Label("🔄 Stride:", style={'color': '#e0e0e0', 'fontWeight': 'bold', 'marginBottom': 3, 'fontSize': '11px'}),
                dcc.Slider(
                    id='stride-slider',
                    min=1, max=10, step=1, value=5,
                    marks={i: {'label': str(i), 'style': {'color': '#e0e0e0', 'fontSize': '9px'}} for i in range(1, 11)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ], style={
            'flex': '0 0 65%',
            'backgroundColor': '#2a2a2a', 'padding': '8px', 'borderRadius': '6px'
        })
    ], style={'display': 'flex', 'alignItems': 'flex-start', 'marginTop': '8px'}),
    
    # Footer compatto
    html.Div([
        html.P("💡 Seleziona diverse immagini test e modifica i parametri per vedere l'effetto in tempo reale",
               style={'textAlign': 'center', 'color': '#666', 'fontSize': '11px', 'margin': '6px 0'})
    ])
], style={'backgroundColor': '#1e1e1e', 'minHeight': '100vh', 'padding': '6px', 'fontFamily': 'Arial'})

# Callback per visualizzare le 20 immagini good
@callback(
    Output('good-images-display', 'children'),
    Input('patch-size-slider', 'value')  # Trigger dummy per caricare una volta
)
def display_good_images(trigger):
    """Visualizza le 20 immagini good come piccole thumbnail"""
    images_html = []
    for i, img in enumerate(good_images):
        # Converte l'immagine in una stringa base64 per la visualizzazione
        img_array = (img[0] * 255).astype(np.uint8)
        images_html.append(
            html.Div([
                html.Img(
                    src=f"data:image/png;base64,{array_to_base64(img_array)}", 
                    style={
                        'width': '100%', 'height': 'auto', 
                        'border': '1px solid #444',
                        'maxWidth': '25px',  # Thumbnail molto piccole
                        'maxHeight': '25px'
                    }
                ),
                html.P(f"G{i+1}", style={
                    'textAlign': 'center', 
                    'fontSize': '6px', 
                    'color': '#4ecdc4', 
                    'margin': '0px',
                    'lineHeight': '1'
                })
            ], style={'textAlign': 'center'})
        )
    return images_html

# Callback per visualizzare le immagini test cliccabili
@callback(
    Output('test-images-display', 'children'),
    Input('selected-test-image', 'data')
)
def display_test_images(selected_index):
    """Visualizza le 10 immagini test cliccabili con highlight per quella selezionata"""
    images_html = []
    for i, (img, label) in enumerate(zip(test_images_mixed, test_labels_mixed)):
        # Converte l'immagine in una stringa base64 per la visualizzazione
        img_array = (img[0] * 255).astype(np.uint8)
        
        # Stile del bordo: evidenzia quella selezionata
        border_color = '#00d4ff' if i == selected_index else '#444'
        border_width = '3px' if i == selected_index else '1px'
        
        color = '#ff6b6b' if label == 'Anomaly' else '#4ecdc4'
        
        images_html.append(
            html.Div([
                html.Img(
                    src=f"data:image/png;base64,{array_to_base64(img_array)}", 
                    style={
                        'width': '100%', 'height': 'auto', 
                        'border': f'{border_width} solid {border_color}',
                        'borderRadius': '3px',
                        'cursor': 'pointer',  # Mostra che è cliccabile
                        'maxWidth': '60px',   # Dimensione moderata per le test images
                        'maxHeight': '60px'
                    },
                    id={'type': 'test-image', 'index': i}  # ID per il click handling
                ),
                html.P(f"T{i+1} ({label[0]})", 
                       style={
                           'textAlign': 'center', 
                           'fontSize': '9px', 
                           'color': color, 
                           'margin': '2px',
                           'fontWeight': 'bold' if i == selected_index else 'normal'
                       })
            ], style={'textAlign': 'center'})
        )
    return images_html

# Callback per gestire i click sulle immagini test
@callback(
    Output('selected-test-image', 'data'),
    Input({'type': 'test-image', 'index': dash.dependencies.ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def handle_test_image_click(n_clicks_list):
    """Gestisce i click sulle immagini test"""
    # Trova quale immagine è stata cliccata
    ctx = dash.callback_context
    if not ctx.triggered:
        return 0
    
    # Estrae l'indice dall'ID del componente cliccato usando JSON parsing
    triggered_id = ctx.triggered[0]['prop_id']
    if 'index' in triggered_id:
        import json
        # Estrae la parte JSON dell'ID
        json_start = triggered_id.find('{')
        json_end = triggered_id.find('}') + 1
        id_dict = json.loads(triggered_id[json_start:json_end])
        return id_dict['index']
    
    return 0

def array_to_base64(img_array):
    """Converte array numpy in stringa base64 per visualizzazione HTML"""
    from PIL import Image
    import io
    import base64
    
    img_pil = Image.fromarray(img_array, mode='L')
    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

# Callback aggiornato con selezione tramite click
@callback(
    [Output('dashboard-plots', 'figure'),
     Output('metrics-display', 'children')],
    [Input('patch-size-slider', 'value'),
     Input('stride-slider', 'value'),
     Input('selected-test-image', 'data')]
)
def update_dashboard(patch_size, stride, test_img_idx):
    """Aggiorna i grafici della dashboard in base ai parametri e immagine selezionata"""
    # Ottieni l'immagine di test selezionata
    current_test_img = test_images_mixed[test_img_idx]
    current_label = test_labels_mixed[test_img_idx]
    
    # Calcola i dati
    anomaly_map, scores, patches_analyzed = calculate_anomaly_map(patch_size, stride, current_test_img)
    
    # Crea subplot con layout migliorato
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'🖼️ Test Image ({current_label})', '🔥 Anomaly Heatmap', 
                       '📊 Distribuzione Scores', '📈 Statistiche Chiave'],
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
               [{'type': 'histogram'}, {'type': 'bar'}]],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )
    
    # 1. Immagine originale
    fig.add_trace(go.Heatmap(
        z=np.flipud(current_test_img[0]),
        colorscale='gray',
        showscale=False,
        hovertemplate='🎯 Posizione: (%{x}, %{y})<br>💾 Valore pixel: %{z:.3f}<extra></extra>'
    ), row=1, col=1)
    
    # 2. Anomaly heatmap con colorbar integrata e scala fissa
    fig.add_trace(go.Heatmap(
        z=np.flipud(anomaly_map),
        colorscale='Blues',
        showscale=True,
        zmin=ANOMALY_SCALE_MIN,  # Scala fissa minima
        zmax=ANOMALY_SCALE_MAX,  # Scala fissa massima
        colorbar=dict(
            title="Anomaly Score",
            len=0.4,  # Altezza della colorbar
            thickness=10,  # Spessore della colorbar
            x=1.02,  # Posizione x (vicino alla heatmap)
            y=0.75   # Posizione y (centrata sulla heatmap)
        ),
        hovertemplate='🎯 Posizione: (%{x}, %{y})<br>🚨 Score: %{z:.4f}<extra></extra>'
    ), row=1, col=2)
    
    # 3. Istogramma scores
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=15,
        marker_color='#636EFA',
        opacity=0.8,
        name='Distribution'
    ), row=2, col=1)
    
    # 4. Statistiche chiave
    stats_names = ['🔻 Min', '📊 Mean', '🔺 Max', '📈 Median']
    stats_values = [min(scores), np.mean(scores), max(scores), np.median(scores)]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    
    fig.add_trace(go.Bar(
        x=stats_names,
        y=stats_values,
        marker_color=colors,
        opacity=0.8,
        text=[f'{v:.4f}' for v in stats_values],
        textposition='auto'
    ), row=2, col=2)
    
    # Layout globale compatto
    fig.update_layout(
        template='plotly_dark',
        title=f"🔬 PatchCore Analysis - Test: {current_label} | Patch: {patch_size}x{patch_size}, Stride: {stride}",
        height=320,
        showlegend=False,
        font=dict(color='#e0e0e0', size=9),
        margin=dict(t=40, b=15, l=25, r=70)
    )
    
    # Nascondi tick per heatmaps e forza aspect ratio quadrato
    fig.update_xaxes(showticklabels=False, row=1, col=1, constrain='domain')
    fig.update_yaxes(showticklabels=False, row=1, col=1, constrain='domain', scaleanchor="x", scaleratio=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2, constrain='domain')
    fig.update_yaxes(showticklabels=False, row=1, col=2, constrain='domain', scaleanchor="x2", scaleratio=1)
    
    # Labels per grafici
    fig.update_xaxes(title_text="Score Value", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=2)
    
    # Metriche in tempo reale
    overlap_pct = max(0, (patch_size - stride) / patch_size * 100) if stride < patch_size else 0
    
    metrics_content = [
        html.H3("📊 Metriche Real-time", style={'color': '#00d4ff', 'marginBottom': 8, 'fontSize': '16px'}),
        html.Div([
            html.Span(f"🎯 Test: ", style={'fontWeight': 'bold', 'fontSize': '13px'}),
            html.Span(f"Img {test_img_idx+1} ({current_label})", 
                     style={'color': '#ff6b6b' if current_label == 'Anomaly' else '#4ecdc4', 'fontSize': '13px'})
        ], style={'margin': '2px 0'}),
        html.Div([
            html.Span(f"🔍 Patches: ", style={'fontWeight': 'bold', 'fontSize': '13px'}),
            html.Span(f"{patches_analyzed}", style={'color': '#4ecdc4', 'fontSize': '13px'})
        ], style={'margin': '2px 0'}),
        html.Div([
            html.Span(f"� Medio: ", style={'fontWeight': 'bold', 'fontSize': '13px'}),
            html.Span(f"{np.mean(scores):.4f}", style={'color': '#45b7d1', 'fontSize': '13px'})
        ], style={'margin': '2px 0'}),
        html.Div([
            html.Span(f"� Range: ", style={'fontWeight': 'bold', 'fontSize': '13px'}),
            html.Span(f"{min(scores):.4f} - {max(scores):.4f}", style={'color': '#96ceb4', 'fontSize': '13px'})
        ], style={'margin': '2px 0'}),
        html.Div([
            html.Span(f"⚙️ Overlap: ", style={'fontWeight': 'bold', 'fontSize': '13px'}),
            html.Span(f"{overlap_pct:.1f}%", style={'color': '#ff6b6b', 'fontSize': '13px'})
        ], style={'margin': '2px 0'})
    ]
    
    return fig, metrics_content

def main():
    """Avvia la dashboard"""
    print("🚀 Avvio PatchCore Dashboard Standalone...")
    print("📊 Features disponibili:")
    print("  ✅ Heatmap interattive con hover dettagliato")
    print("  ✅ Distribuzione scores in tempo reale")
    print("  ✅ Statistiche dinamiche")
    print("  ✅ Controlli slider responsive")
    print("  ✅ Tema scuro professionale")
    print()
    print("🌐 Dashboard disponibile su: http://localhost:8051")
    print("🛑 Per fermare: Ctrl+C")
    print()
    
    try:
        app.run(debug=False, host='127.0.0.1', port=8051)
    except KeyboardInterrupt:
        print("\n👋 Dashboard fermata dall'utente")
    except Exception as e:
        print(f"❌ Errore: {e}")

if __name__ == "__main__":
    main()
