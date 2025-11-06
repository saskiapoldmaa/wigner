#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wigner viewer with editable α and dynamic slider ticks:
  - Default ticks: (-10, +10)
  - If user enters large α, slider auto-expands and updates ticks dynamically (correct side)
  - Expression input works on Enter
  - Robust against None while typing / dragging
"""

import math
import re
import numpy as np
from qutip import destroy, qeye, basis, wigner
from dash import Dash, html, dcc, ctx
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.exceptions import PreventUpdate


# ---------------------------------------------------------
# App setup
# ---------------------------------------------------------
app = Dash(
    __name__,
    external_scripts=[
    {'src': 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js'}
    ]
)
app.title = "Wigner Viewer"
app.index_string = '''
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <script>
      // MathJax v3 config (must be defined BEFORE the script loads)
      window.MathJax = {
        tex: {
          inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
          displayMath: [['$$','$$']],
          processEscapes: true
        },
        options: {
          skipHtmlTags: ['script','noscript','style','textarea','pre','code']
        },
        svg: { fontCache: 'global' }
      };
    </script>
    <style>
    @keyframes blink {
    0%   { opacity: 0.2; }
    50%  { opacity: 1; }
    100% { opacity: 0.2; }
    }
    #loading-text::after {
    content: "...";
    animation: blink 1.2s infinite;
    }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
      <script>
        // Re-typeset whenever Dash mutates the DOM
        (function(){
          function typeset(){ if (window.MathJax && MathJax.typesetPromise) { MathJax.typesetPromise(); } }
          const root = document.getElementById('react-entry-point') || document.body;
          const obs = new MutationObserver(typeset);
          obs.observe(root, {childList: true, subtree: true});
          typeset(); // initial
        })();
      </script>
    </footer>
  </body>
</html>
'''


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
import re

def expr_to_latex(expr):
    print("THe expr is", expr)

    """Convert operator expression to LaTeX format (robust against a^\dagger**2, alpha^*, etc.)."""
    if not expr:
        return ""

    # 0) normalize '**' to '^' so we only handle one exponent marker
    s = expr.replace("**", "^")
    chars = list(s)
    indices = [i for i, x in enumerate(chars) if x == "^"]
    for i in indices:
        chars.insert(i, "}")
        left_brace_id = i
        while chars[left_brace_id] != " " and chars[left_brace_id] != "(" and left_brace_id>0:
            left_brace_id -= 1
        chars.insert(left_brace_id, "{")
                    
    s = "".join(chars)
    print("Check -1:", s)

    # 2) pretty operator 'a' -> \hat{a} (but keep other words intact)
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == 'a':
            prev_ok = (i == 0) or (not s[i-1].isalnum())
            next_is_word = (i+1 < len(s) and s[i+1].isalnum())
            if prev_ok and (not next_is_word):
                out.append('a')
                i += 1
                continue
        out.append(ch)
        i += 1
    latex = ''.join(out)
    latex = latex.replace("*", " ")

    print("Check 0:", latex)

    # 3) parameters
    
    latex = latex.replace("alpha", r"\alpha")
    latex = latex.replace("alphastar", r"alpha^* ")
    print("Check 1:", latex)

    # replace standalone 'a' with \hat{a}
    latex = re.sub(r'\ba\b', r'\\hat{a}', latex)

    # fix adagger
    latex = latex.replace('adagger', r'\hat{a}^{\dagger}')
    latex = latex.replace('adag', r'\hat{a}^{\dagger}')

    print("latex is", latex)

    # 4) Convert exp(...) -> e^{...} (right-to-left so nested exps work)
    while "exp(" in latex:
        pos = latex.rfind("exp(")
        start = pos + 4
        depth, j = 1, start
        while j < len(latex) and depth > 0:
            if latex[j] == '(':
                depth += 1
            elif latex[j] == ')':
                depth -= 1
            j += 1
        if depth == 0:
            inner = latex[start:j-1]
            latex = f"{latex[:pos]}e^{{{inner}}}{latex[j:]}"
        else:
            break

    print('check 2', latex)
    # Clean spacing a bit
    latex = ' '.join(latex.split())

    # Use display math to avoid inline baseline artifacts
    return f"$ {latex}\\,|0\\rangle $"


def find_matching_bracket(s, start):
    """Find the matching closing bracket for the opening bracket at position start."""
    count = 1
    for i in range(start + 1, len(s)):
        if s[i] == '{':
            count += 1
        elif s[i] == '}':
            count -= 1
            if count == 0:
                return i
    return -1


def num(v, fallback=0.0):
    """Coerce v to a finite float; fallback if v is None/NaN/inf/non-numeric."""
    try:
        if v is None:
            return float(fallback)
        x = float(v)
        if math.isfinite(x):
            return x
        return float(fallback)
    except Exception:
        return float(fallback)

# ---------------------------------------------------------
# Operator Parsing
# ---------------------------------------------------------
def parse_physics_operator(expr: str, N: int = 20, alpha: complex = 0):
    expr = expr.strip()
    expr = expr.replace("†", "dagger").replace("^", "**").replace("−", "-").replace("α", "alpha")

    # --- normalize spacing and implicit multiplications ---
    # add * between symbol-letter or parenthesis transitions, e.g., 'adag a' → 'adag*a'
    expr = re.sub(r'(?<=[a-zA-Z0-9)])\s+(?=[a-zA-Z(])', '*', expr)

    # also handle e.g. ')(' -> ')*('
    expr = re.sub(r'(?<=\))(?=\()', ')*(', expr)

    # --- safe replacements ---
    expr = expr.replace("adagger", "adag")
    expr = expr.replace("alphastar", "np.conjugate(alpha)")

    a = destroy(N)
    adag = a.dag()
    I = qeye(N)

    allowed = {
        "a": a, "adag": adag, "I": I, "alpha": alpha, "np": np,
        "exp": lambda x: x.expm() if hasattr(x, "expm") else np.exp(x),
        "sin": lambda x: x.sinm() if hasattr(x, "sinm") else np.sin(x),
        "cos": lambda x: x.cosm() if hasattr(x, "cosm") else np.cos(x),
    }

    try:
        return eval(expr, {"__builtins__": None}, allowed)
    except Exception as e:
        print(f"⚠️ Parse error for expression: {expr}")
        print(e)
        # return identity as safe fallback
        return qeye(N)



def estimate_N(expr: str, alpha: complex) -> int:
    """
    Estimate a suitable Hilbert space truncation N based on the operator expression
    and the coherent amplitude α (or any parameter magnitude).
    """
    # --- base dimension ---
    N = 50

    # --- increase with |alpha| ---
    N += int(3 * abs(alpha)**2)  # coherent state occupation ~ |α|^2

    # --- increase with operator complexity ---
    # count ladder operators and powers
    num_a = len(re.findall(r'\ba\b', expr))
    num_adag = len(re.findall(r'adag', expr)) + len(re.findall(r'adagger', expr))
    num_pow = len(re.findall(r'\*\*', expr)) + len(re.findall(r'\^', expr))

    complexity = num_a + num_adag + 2 * num_pow
    N += 10 * complexity

    # --- clamp ---
    N = min(max(N, 20), 150)  # between 20 and 150 for stability
    return N

# ---------------------------------------------------------
# Wigner calculation
# ---------------------------------------------------------

def compute_wigner(expr, alpha_val=0+0j, N=None, omega=1.0, xlim=None, points=100):
    if N is None:
        N = estimate_N(expr, alpha_val)
    
    if xlim is None:
        xlim = max(5, 1.5 * abs(alpha_val))
        
    x = np.linspace(-xlim, xlim, points)
    p = np.linspace(-xlim, xlim, points)
    X, P = np.meshgrid(x, p)

    # --- Build the state ---
    # --- Build the state ---
    gnd = basis(N, 0)
    op = parse_physics_operator(expr, N, alpha_val)
    psi_raw = op * gnd

    # --- avoid zero-norm crashes ---
    if psi_raw.norm() < 1e-12:
        psi0 = gnd  # fallback to vacuum
    else:
        psi0 = psi_raw.unit()


    # --- Compute Wigner function ---
    W = wigner(psi0, x, p)

    # --- Quadrature operators ---
    a = destroy(N)
    adag = a.dag()
    x_op = (a + adag) / np.sqrt(2)
    p_op = 1j * (adag - a) / np.sqrt(2)

    # --- Expectation values ---
    def expect_val(op):
        val = (psi0.dag() * op * psi0)
        # It can be Qobj(1x1) or a scalar complex
        return np.real(val.full()[0, 0]) if hasattr(val, "full") else np.real(val)

    x_mean = expect_val(x_op)
    p_mean = expect_val(p_op)

    dx2 = expect_val((x_op - x_mean) ** 2)
    dp2 = expect_val((p_op - p_mean) ** 2)

    delta_x = np.sqrt(dx2)
    delta_p = np.sqrt(dp2)
    product = delta_x * delta_p / 0.5  # in units of ħ/2 (ħ = 1)

    # --- Calculate marginal probabilities ---
    # Project and normalize marginals
    psi_x = np.sum(W, axis=0)  # Projection onto x-axis (marginal in p)
    psi_p = np.sum(W, axis=1)  # Projection onto p-axis (marginal in x)
    
    # Get the maximum height of the Wigner function
    wigner_min = np.min(W)
    wigner_max = np.max(np.abs(W))
    
    # Scale marginals to match the height of the Wigner function
        # --- Calculate marginal probabilities ---
    psi_x = np.sum(W, axis=0)
    psi_p = np.sum(W, axis=1)

    # Normalize and shift to always sit on z=0 plane
    psi_x = psi_x - np.min(psi_x)
    psi_p = psi_p - np.min(psi_p)
    if np.max(np.abs(psi_x)) > 0:
        psi_x /= np.max(np.abs(psi_x))
    if np.max(np.abs(psi_p)) > 0:
        psi_p /= np.max(np.abs(psi_p))

    # Scale to match height of Wigner function
    psi_x *= np.max(np.abs(W))
    psi_p *= np.max(np.abs(W))

    # --- Plot projections ---
    fig = go.Figure()

    # Wigner surface
    fig.add_trace(go.Surface(
        x=X, y=P, z=W,
        colorscale='Viridis',
        showscale=True,
        opacity=1.0,
        cmin=np.min(W),
        cmax=np.max(W),
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True),
        colorbar=dict(title=dict(text='W(x,p)', side='right'))
    ))

    # Add |ψ(x)|² projection on the back wall (y = -xlim)
    mid_x = len(x) // 2
    text_x = [''] * len(x)
    text_x[mid_x] = '|ψ(x)|²'

    fig.add_trace(go.Scatter3d(
        x=x,
        y=np.full_like(x, xlim * 0.95),  # near back wall
        z=psi_x,
        mode='lines+text',
        line=dict(color='#1f77b4', width=3),
        text=text_x,
        textposition='top center',
        textfont=dict(color='#1f77b4', size=12),
        showlegend=False
    ))

    # Add |ψ(p)|² projection on the right wall (x = xlim)
    mid_p = len(p) // 2
    text_p = [''] * len(p)
    text_p[mid_p] = '|ψ(p)|²'

    fig.add_trace(go.Scatter3d(
        x=np.full_like(p, xlim * 0.95),  # near right wall
        y=p,
        z=psi_p,
        mode='lines+text',
        line=dict(color='#7ad151', width=3),
        text=text_p,
        textposition='top center',
        textfont=dict(color='#7ad151', size=12),
        showlegend=False
    ))

    
    # Add floor grid
    fig.add_trace(go.Surface(
        x=X,
        y=P,
        z=np.full_like(X, -0.1),  # Slightly below z=0
        colorscale=[[0, 'rgba(0,0,0,0.1)'], [1, 'rgba(0,0,0,0.1)']],
        showscale=False,
        opacity=0.3,
        surfacecolor=np.ones_like(X) * 0.5
    ))
    
    # Update layout for 3D view
    fig.update_layout(
        title= f"Uncertainty <br>ΔxΔp = {product:.3f} × (ħ/2)",
        scene=dict(
            xaxis_title="⟨x⟩",
            yaxis_title="⟨p⟩",
            zaxis_title="W(x,p)",
            aspectratio=dict(x=1, y=1, z=0.5),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            ),
            xaxis=dict(
                range=[-xlim, xlim],
                showbackground=False,
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                zerolinecolor='rgba(0,0,0,0.5)'
            ),
            yaxis=dict(
                range=[-xlim, xlim],
                showbackground=False,
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                zerolinecolor='rgba(0,0,0,0.5)'
            ),
            zaxis=dict(
                range=[-0.1, wigner_max * 1.1],  # Auto-scale based on Wigner function height
                showbackground=False,
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                zerolinecolor='rgba(0,0,0,0.5)'
            )
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    return fig

# ---------------------------------------------------------
# Helper to auto-adjust slider span (expand correct side)
# ---------------------------------------------------------
def adjust_dynamic_bounds(v, default_min=-1.0, default_max=1.0):
    """Expand slider range and tick labels dynamically, keeping visible end ticks."""
    v = num(v, 0.0)

    mn, mx = default_min, default_max
    if v < mn:  # large negative → expand lower bound
        mn = min(v * 1.1, default_min)  # extend downward by ~10%
    elif v > mx:  # large positive → expand upper bound
        mx = max(v * 1.1, default_max)

    # Use string keys so ticks never disappear
    marks = {str(int(mn)): str(int(mn)), str(int(mx)): str(int(mx))}
    return float(mn), float(mx), marks

# ---------------------------------------------------------
# Layout
# ---------------------------------------------------------
app.layout = html.Div([
    html.Div([
        html.H3("Operator:", style={"marginTop": "0", "color": "#2c3e50"}),
        html.Div([
            html.Div(
                dcc.Input(
                    id="expr-input",
                    type="text",
                    placeholder="Enter operator (e.g. exp(a alpha - astar adagger))",
                    debounce=False,
                    style={
                        "width": "100%",
                        "height": "40px",
                        "fontFamily": "monospace",
                        "padding": "8px 12px",
                        "border": "1px solid #ddd",
                        "borderRadius": "4px",
                        "boxSizing": "border-box"
                    },
                ),
                id="expr-input-container",
                n_clicks=0,
                style={"position": "relative"}
            ),
            html.Div(id="expr-suggestions-box", style={
                "position": "relative",
                "width": "100%"
            }),
            dcc.Store(id="show-suggestions", data=False),
            html.Div(id="overlay-click-capture", n_clicks=0, style={
                "position": "fixed",
                "top": 0, "left": 0,
                "width": "100vw", "height": "100vh",
                "zIndex": 5,
                "display": "none"
            }),
        ], style={"position": "relative"}),

        html.Button("Add Expression", 
            id="add-btn", 
            n_clicks=0,
            style={
                "marginTop": "15px",
                "width": "100%",
                "padding": "8px",
                "backgroundColor": "#4a90e2",
                "color": "white",
                "border": "none",
                "borderRadius": "4px",
                "cursor": "pointer"
            }
        ),
        html.Hr(style={"margin": "20px 0", "borderColor": "#eee"}),
        html.Div(id="expr-list", children=[], style={"overflowY": "auto"}),
    ], style={
        "flex": "0 0 25%", 
        "padding": "15px", 
        "backgroundColor": "#f8f9fa",
        "height": "100vh", 
        "boxShadow": "2px 0px 6px rgba(0,0,0,0.1)",
        "overflowY": "auto",
        "boxSizing": "border-box"
    }),

    html.Div([
        dcc.Loading(
            id="loading-container",
            type="default",  # spinner type ("default", "circle", "dot", etc.)
            children=dcc.Graph(
                id="wigner-graph",
                style={"height": "100vh", "width": "100%"}
            ),
            style={
                "position": "absolute",
                "top": "0",
                "left": "0",
                "width": "100%",
                "height": "100%",
                "display": "flex",
                "alignItems": "flex-start",
                "justifyContent": "center",
                "paddingTop": "10px"
            }
        ),
        html.Div(
            id="loading-text",
            style={
                "position": "absolute",
                "top": "10px",
                "left": "50%",
                "transform": "translateX(-50%)",
                "fontSize": "20px",
                "fontWeight": "bold",
                "color": "#333",
                "animation": "blink 1.2s infinite",
                "display": "none"  # hidden because dcc.Loading handles visibility
            }
        )
    ], style={"flex": "1", "backgroundColor": "#fff", "position": "relative"}),



], style={
    "display": "flex", 
    "flexDirection": "row", 
    "width": "100vw", 
    "height": "100vh",
    "margin": "0",
    "padding": "0",
    "fontFamily": "Arial, sans-serif"
})

# ---------------------------------------------------------
# Add expression (Enter or button)
# ---------------------------------------------------------
@app.callback(
    Output("overlay-click-capture", "style"),
    Input("show-suggestions", "data")
)
def toggle_overlay(visible):
    return {
        "position": "fixed",
        "top": 0, "left": 0,
        "width": "100vw", "height": "100vh",
        "zIndex": 5,
        "display": "block" if visible else "none"
    }

@app.callback(
    Output("expr-suggestions-box", "children"),
    Output("show-suggestions", "data"),
    Input("expr-input-container", "n_clicks"),   # clicking input shows suggestions
    Input("expr-input", "n_submit"),             # hide on Enter
    Input("add-btn", "n_clicks"),                # hide on Add
    Input("overlay-click-capture", "n_clicks"),  # hide on outside click
    State("show-suggestions", "data"),
    prevent_initial_call=True,
)
def show_suggestions(input_clicks, enter_pressed, add_clicks, overlay_clicks, visible):
    trig_prop = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    # Hide suggestions on Enter, Add, or clicking outside
    if trig_prop in (
        "expr-input.n_submit",
        "add-btn.n_clicks",
        "overlay-click-capture.n_clicks",
    ):
        return [], False

    # Show all suggestions when clicking inside the input
    if trig_prop == "expr-input-container.n_clicks":
        visible = True

    # If dropdown not visible, return nothing
    if not visible:
        return [], False

    # Always show full list (no filtering)
    suggestions = [
        {"label": "D(α)", "value": "exp(alpha*adagger - alphastar*a)"},
        {"label": "S(α)", "value": "exp(0.5*(alpha*adagger**2 - alphastar*a**2))"},
        {"label": "D(α)S(α)", "value": "exp(alpha*adagger - alphastar*a)*exp(0.5*(alpha*adagger**2 - alphastar*a**2))"},
        {"label": "S(α)D(α)", "value": "exp(0.5*(alpha*adagger**2 - alphastar*a**2))*exp(alpha*adagger - alphastar*a)"},
        {"label": "D(α)+D(-α)", "value": "exp(-alpha*adagger + alphastar*a) + exp(alpha*adagger - alphastar*a)"},
    ]

    return html.Ul([
        html.Li(
            s["label"],
            id={"type": "suggestion-item", "value": s["value"]},
            n_clicks=0,
            style={
                "listStyle": "none",
                "padding": "6px 10px",
                "cursor": "pointer",
                "backgroundColor": "#fff",
                "borderBottom": "1px solid #ddd"
            }
        )
        for s in suggestions
    ], style={
        "position": "absolute",
        "top": "40px",
        "left": "0",
        "right": "0",
        "backgroundColor": "white",
        "border": "1px solid #ccc",
        "zIndex": 10,
        "margin": 0,
        "padding": 0,
        "maxHeight": "150px",
        "overflowY": "auto",
        "boxShadow": "0 2px 6px rgba(0,0,0,0.1)"
    }), True


@app.callback(
    Output("expr-input", "value"),
    Output("expr-suggestions-box", "children", allow_duplicate=True),
    Output("show-suggestions", "data", allow_duplicate=True),
    Output("add-btn", "n_clicks"),
    Input({"type": "suggestion-item", "value": ALL}, "n_clicks"),
    State({"type": "suggestion-item", "value": ALL}, "id"),
    State("add-btn", "n_clicks"),
    prevent_initial_call=True,
)
def click_suggestion(n_clicks, ids, n_clicks_btn):
    if not n_clicks or not ids:
        raise PreventUpdate
    for i, n in enumerate(n_clicks):
        if n and ids[i]:
            # ✅ 4 return values, matching all Outputs
            return (
                ids[i]["value"],  # expr-input.value
                [],               # expr-suggestions-box.children (clear menu)
                False,            # show-suggestions.data (hide dropdown)
                (n_clicks_btn or 0) + 1  # add-btn.n_clicks (simulate Add click)
            )
    raise PreventUpdate


@app.callback(
    Output("expr-list", "children"),
    Input("add-btn", "n_clicks"),
    Input("expr-input", "n_submit"),   # Enter key
    State("expr-input", "value"),
    State("expr-list", "children"),
    prevent_initial_call=True
)
def add_expression(n_clicks, n_submit, expr, current_list):
    """Replace the sidebar with only the most recent expression."""
    if not expr:
        return []

    expr_idx = 0  # always index 0 since we only keep one expression
    expr_stripped = expr.strip()

    # Store the original expression in a hidden div
    children = [
        html.Div(expr_stripped, id={"type": "expr-text", "index": expr_idx}, style={"display": "none"}),
        html.Div(
            expr_to_latex(expr_stripped),
            style={
                "display": "block",
                "lineHeight": "1.4em",
                "textAlign": "center",
                "overflow": "visible",
                "fontSize": "1.3em"
            }
        )
    ]

    # ✅ Only add α controls if α appears
    has_alpha = bool(re.search(r"(alpha|α)", expr_stripped))
    if has_alpha:
        children.append(html.Div([
            html.Div([
                html.Label("Re(α) =", style={"marginRight": "6px"}),
                dcc.Input(
                    id={"type": "alpha-input-re", "index": expr_idx},
                    type="number", step=0.01, value=0.0,
                    style={"width": "80px", "marginRight": "8px"}
                ),
            ], style={"display": "flex", "alignItems": "center", "gap": "6px",
                      "marginBottom": "4px"}),

            dcc.Slider(
                id={"type": "alpha-slider-re", "index": expr_idx},
                min=-2, max=2, step=0.01, value=0.0,
                marks={-2: "-2", 2: "2"},
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="drag"
            ),
            html.Br(),

            html.Div([
                html.Label("Im(α) =", style={"marginRight": "6px"}),
                dcc.Input(
                    id={"type": "alpha-input-im", "index": expr_idx},
                    type="number", step=0.01, value=0.0,
                    style={"width": "80px", "marginRight": "8px"}
                ),
            ], style={"display": "flex", "alignItems": "center", "gap": "6px",
                      "marginBottom": "4px"}),

            dcc.Slider(
                id={"type": "alpha-slider-im", "index": expr_idx},
                min=-2, max=2, step=0.01, value=0.0,
                marks={-2: "-2", 2: "2"},
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="drag"
            ),
        ], style={"marginBottom": "25px"}))

    # Replace the sidebar content entirely
    return [html.Div(children)]


# ---------------------------------------------------------
# Synchronize α inputs and sliders + dynamic ticks + plot
# ---------------------------------------------------------
@app.callback(
    Output({"type": "alpha-input-re", "index": ALL}, "value"),
    Output({"type": "alpha-input-im", "index": ALL}, "value"),
    Output({"type": "alpha-slider-re", "index": ALL}, "value"),
    Output({"type": "alpha-slider-im", "index": ALL}, "value"),
    Output({"type": "alpha-slider-re", "index": ALL}, "min"),
    Output({"type": "alpha-slider-re", "index": ALL}, "max"),
    Output({"type": "alpha-slider-re", "index": ALL}, "marks"),
    Output({"type": "alpha-slider-im", "index": ALL}, "min"),
    Output({"type": "alpha-slider-im", "index": ALL}, "max"),
    Output({"type": "alpha-slider-im", "index": ALL}, "marks"),
    Output("wigner-graph", "figure"),

    # 🔹 Trigger both alpha controls and expression changes
    Input({"type": "alpha-input-re", "index": ALL}, "value"),
    Input({"type": "alpha-input-im", "index": ALL}, "value"),
    Input({"type": "alpha-slider-re", "index": ALL}, "value"),
    Input({"type": "alpha-slider-im", "index": ALL}, "value"),
    Input("expr-list", "children"),  # ✅ new input trigger

    State("wigner-graph", "figure"),
)
def sync_all(re_inputs, im_inputs, re_sliders, im_sliders, expr_list, prev_fig):
    """Return clean outputs even when there are no α controls."""
    # helper for empty pattern outputs (must be lists)
    def empty_outputs():
        return [ [] for _ in range(10) ]

    if not expr_list:
        return empty_outputs() + [go.Figure()]

    # extract latest expression text
    expr_text = None
    for child in expr_list[-1]["props"]["children"]:
        if isinstance(child, dict) and child.get("type") == "Div":
            props = child.get("props", {})
            if props.get("id") and props["id"].get("type") == "expr-text":
                expr_text = props["children"]
                break
    if not isinstance(expr_text, str) or not expr_text.strip():
        expr_text = "a"

    has_alpha = bool(re.search(r"(alpha|α)", expr_text))

    # preserve camera
    camera = None
    if prev_fig and "layout" in prev_fig and "scene" in prev_fig["layout"]:
        camera = prev_fig["layout"]["scene"].get("camera")

    # case 1: no α → simple plot, empty control lists
    if not has_alpha:
        fig = compute_wigner(expr_text, 0.0)
        if camera:
            fig.update_layout(scene_camera=camera)
        return empty_outputs() + [fig]

    # case 2: α present → normal handling
    re_val = num(re_sliders[0] if re_sliders else 0.0)
    im_val = num(im_sliders[0] if im_sliders else 0.0)
    fig = compute_wigner(expr_text, re_val + 1j * im_val)
    if camera:
        fig.update_layout(scene_camera=camera)

    # valid single-value lists for each pattern output
    re_min, re_max, im_min, im_max = -2, 2, -2, 2
    re_marks = {-2: "-2", 2: "2"}
    im_marks = {-2: "-2", 2: "2"}

    return (
        [re_val], [im_val],
        [re_val], [im_val],
        [re_min], [re_max], [re_marks],
        [im_min], [im_max], [im_marks],
        fig
    )



# ---------------------------------------------------------
# Show "Calculating..." as soon as any input triggers
@app.callback(
    Output("loading-text", "style"),
    Input({"type": "alpha-input-re", "index": ALL}, "value"),
    Input({"type": "alpha-input-im", "index": ALL}, "value"),
    Input({"type": "alpha-slider-re", "index": ALL}, "value"),
    Input({"type": "alpha-slider-im", "index": ALL}, "value"),
    prevent_initial_call=True
)
def show_loading(*_):
    return {
        "position": "absolute",
        "top": "10px",
        "left": "50%",
        "transform": "translateX(-50%)",
        "fontSize": "20px",
        "fontWeight": "bold",
        "color": "#333",
        "display": "block"
    }


# Hide "Calculating..." once figure update is complete
@app.callback(
    Output("loading-text", "style", allow_duplicate=True),
    Input("wigner-graph", "figure"),
    prevent_initial_call=True
)
def hide_loading(_):
    return {"display": "none"}


if __name__ == "__main__":
    app.run(debug=True)
