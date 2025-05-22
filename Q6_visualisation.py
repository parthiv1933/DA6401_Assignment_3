import numpy as np
import time
from IPython.display import display, HTML, clear_output

def format_colored_text(char, color='black'):
    """Returns an HTML-formatted string with background color for a character."""
    if char == ' ':
        return f"<text style='color:#000;padding-left:10px;background-color:{color}'> </text>"
    return f"<text style='color:#000;background-color:{color}'>{char} </text>"

def render_colored_sequence(pairs):
    """Displays a sequence of (char, color) pairs as HTML."""
    html_sequence = ''.join([format_colored_text(ch, bg) for ch, bg in pairs])
    display(HTML(html_sequence))

def get_heatmap_color(value):
    """Maps a normalized attention score to a background color."""
    palette = [
        '#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF', '#f9e8e8', '#f9e8e8',
        '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f', '#f68f8f', '#f47676',
        '#f45f5f', '#f45f5f', '#f34343', '#f34343', '#f33b3b', '#f33b3b',
        '#f33b3b', '#f42e2e', '#f42e2e'
    ]
    index = min(int((value * 100) / 5), len(palette) - 1)
    return palette[index]

def display_attention(attn_matrix, predicted_tokens, sample_index, input_tokens):
    """
    Visualizes the attention weights between input tokens and predicted tokens.
    
    Parameters:
    - attn_matrix: list of attention vectors for each predicted token
    - predicted_tokens: list of generated output tokens
    - sample_index: index of the input sentence to visualize
    - input_tokens: list of input sentences (tokenized)
    """
    trimmed_weights = [layer[:len(input_tokens[sample_index])] for layer in attn_matrix[:-1]]
    attn_array = np.array(trimmed_weights)
    
    max_weight = np.max(attn_array)
    identity_overlay = np.identity(max(attn_array.shape)) * max_weight

    for t in range(len(attn_array)):
        token_colors = [(input_tokens[sample_index][k], get_heatmap_color(attn_array[t][k])) 
                        for k in range(attn_array.shape[1])]
        pred_colors = [(predicted_tokens[k], get_heatmap_color(identity_overlay[t][k])) 
                       for k in range(attn_array.shape[0])]
        
        clear_output(wait=True)
        render_colored_sequence(pred_colors)
        render_colored_sequence(token_colors)
        time.sleep(2)
