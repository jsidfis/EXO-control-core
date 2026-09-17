import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.models import LSTMBaseline, GRUBaseline, TransformerBaseline

BASELINE_REGISTRY = {
    'lstm_last': LSTMBaseline,
    'gru_last': GRUBaseline,
    'transformer_last': TransformerBaseline,
}


def create_baseline_model(model_name, config):
    if model_name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {model_name}. Available: {list(BASELINE_REGISTRY.keys())}")

    model_cls = BASELINE_REGISTRY[model_name]

    common_kwargs = {
        'input_dim': config['input_dim'],
        'output_dim': config.get('output_dim', 4),
    }

    if model_name in ('lstm_last', 'gru_last'):
        common_kwargs.update({
            'hidden_dim': config.get('hidden_dim', 128),
            'num_layers': config.get('num_layers', 2),
            'dropout': config.get('dropout', 0.3),
            'bidirectional': config.get('bidirectional', True),
        })
    elif model_name == 'transformer_last':
        common_kwargs.update({
            'd_model': config.get('d_model', 128),
            'nhead': config.get('nhead', 4),
            'num_layers': config.get('num_layers', 4),
            'dim_feedforward': config.get('dim_feedforward', 256),
            'dropout': config.get('dropout', 0.1),
        })

    return model_cls(**common_kwargs)
