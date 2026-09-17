from baselines.models import LSTMBaseline, GRUBaseline, TransformerBaseline
from models.ablation_models import TCNOnlyLast, SSMOnlyLast
from models.enhanced_tcn import EnhancedTCNLast
from models.parallel_tcn_ssm import ParallelTCNSSMLast


MODEL_REGISTRY = {
    'parallel_tcn_ssm_last': ParallelTCNSSMLast,
    'enhanced_tcn_last': EnhancedTCNLast,
    'tcn_only_last': TCNOnlyLast,
    'ssm_only_last': SSMOnlyLast,
    'lstm_last': LSTMBaseline,
    'gru_last': GRUBaseline,
    'transformer_last': TransformerBaseline,
}


def create_model(model_name, config):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    common_kwargs = {
        'input_dim': config['input_dim'],
        'output_dim': config.get('output_dim', 4),
        'num_channels': config.get('num_channels', [64, 64, 64]),
        'kernel_size': config.get('kernel_size', 7),
        'dropout': config.get('dropout', 0.4),
    }

    if model_name == 'parallel_tcn_ssm_last':
        common_kwargs.update({
            'ssm_branch_layers': config.get('ssm_branch_layers', 1),
            'ssm_dropout': config.get('ssm_dropout', 0.1),
            'ssm_scale_init': config.get('ssm_scale_init', 0.1),
            'ssm_expand': config.get('ssm_expand', 2),
            'ssm_d_state': config.get('ssm_d_state', 8),
            'fusion': config.get('fusion', 'gated'),
            'head_type': config.get('head_type', 'linear'),
            'head_hidden_dim': config.get('head_hidden_dim', 64),
            'head_dropout': config.get('head_dropout', 0.0),
            'use_angle_only_swing': config.get('use_angle_only_swing', False),
            'angle_feature_dim': config.get('angle_feature_dim', 12),
        })
    elif model_name == 'ssm_only_last':
        common_kwargs.update({
            'ssm_branch_layers': config.get('ssm_branch_layers', 1),
            'ssm_dropout': config.get('ssm_dropout', 0.1),
            'ssm_scale_init': config.get('ssm_scale_init', 0.1),
            'ssm_expand': config.get('ssm_expand', 2),
            'ssm_d_state': config.get('ssm_d_state', 8),
        })
    elif model_name in ('lstm_last', 'gru_last'):
        common_kwargs = {
            'input_dim': config['input_dim'],
            'output_dim': config.get('output_dim', 4),
            'hidden_dim': config.get('hidden_dim', 128),
            'num_layers': config.get('num_layers', 2),
            'dropout': config.get('dropout', 0.3),
            'bidirectional': config.get('bidirectional', True),
        }
    elif model_name == 'transformer_last':
        common_kwargs = {
            'input_dim': config['input_dim'],
            'output_dim': config.get('output_dim', 4),
            'd_model': config.get('d_model', 128),
            'nhead': config.get('nhead', 4),
            'num_layers': config.get('num_layers', 4),
            'dim_feedforward': config.get('dim_feedforward', 256),
            'dropout': config.get('dropout', 0.1),
        }

    return MODEL_REGISTRY[model_name](**common_kwargs)
