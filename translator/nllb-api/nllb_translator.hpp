struct ModelConfig {
    int hidden_size;
    int num_heads;
    int num_layers;
    
    ModelConfig(int hidden_size = 1024, int num_heads = 16, int num_layers = 12)
        : hidden_size(hidden_size), num_heads(num_heads), num_layers(num_layers) {}
}; 