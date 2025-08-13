#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000
#define BATCH_SIZE 32
#define EPOCHS 20
#define LEARNING_RATE 0.001f

typedef struct {
    float *weights1, *weights2;
    float *bias1, *bias2;
    float *grad_weights1, *grad_weights2;
    float *grad_bias1, *grad_bias2;
    cublasHandle_t handle;
} CuBLASNeuralNetwork;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

void initialize_weights(float *weights, int size) {
    float scale = sqrtf(2.0f / INPUT_SIZE); // Xavier/He init
    for (int i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

__global__ void relu_kernel(float *x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void bias_add_kernel(float *x, float *bias, int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / size;
    int i = idx % size;

    if (b < batch_size && i < size) {
        x[idx] += bias[i];
    }
}

__global__ void softmax_kernel(float *x, int batch_size, int size) {
    int b = blockIdx.x;
    if (b < batch_size) {
        float max_val = x[b * size];
        for (int i = 1; i < size; ++i) {
            max_val = fmaxf(max_val, x[b * size + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            x[b * size + i] = expf(x[b * size + i] - max_val);
            sum += x[b * size + i];
        }

        for (int i = 0; i < size; ++i) {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
        }
    }
}

__global__ void compute_output_gradient(float *output, int *labels, float *grad_output, int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (idx < output_size && batch_idx < batch_size) {
        int index = batch_idx * output_size + idx;
        grad_output[index] = output[index];
        if (idx == labels[batch_idx]) {
            grad_output[index] -= 1.0f;
        }
    }
}

__global__ void compute_bias_gradient(float *grad_bias, float *grad, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum += grad[i * size + idx];
        }
        grad_bias[idx] = sum;
    }
}

__global__ void update_weights_kernel(float *weights, float *grad_weights, int size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * grad_weights[idx];
    }
}

void initialize_neural_network(CuBLASNeuralNetwork *nn) {
    CUDA_CHECK(cudaMalloc(&nn->weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_bias2, OUTPUT_SIZE * sizeof(float)));

    float *h_weights1 = (float *)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float *h_weights2 = (float *)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *h_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    float *h_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    initialize_weights(h_weights1, HIDDEN_SIZE * INPUT_SIZE);
    initialize_weights(h_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    initialize_bias(h_bias1, HIDDEN_SIZE);
    initialize_bias(h_bias2, OUTPUT_SIZE);

    CUDA_CHECK(cudaMemcpy(nn->weights1, h_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->weights2, h_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias1, h_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias2, h_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    free(h_weights1);
    free(h_weights2);
    free(h_bias1);
    free(h_bias2);

    CUBLAS_CHECK(cublasCreate(&nn->handle));
}

void forward_cublas(CuBLASNeuralNetwork *nn, float *d_input, float *d_hidden, float *d_output, int batch_size) {
    const float alpha = 1.0f, beta = 0.0f;

    // Layer 1: input @ weights1.T + bias1
    // d_hidden = d_input @ d_weights1.T (batch_size x input_size) @ (input_size x hidden_size) = (batch_size x hidden_size)
    CUBLAS_CHECK(cublasSgemm(nn->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            HIDDEN_SIZE, batch_size, INPUT_SIZE,
                            &alpha,
                            nn->weights1, INPUT_SIZE,    // weights1 = (hidden_size x input_size), transpose -> (input_size x hidden_size)
                            d_input, INPUT_SIZE,         // input = (batch_size x input_size)
                            &beta,
                            d_hidden, HIDDEN_SIZE));     // output = (batch_size x hidden_size)

    bias_add_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    relu_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Layer 2: hidden @ weights2.T + bias2  
    // d_output = d_hidden @ d_weights2.T (batch_size x hidden_size) @ (hidden_size x output_size) = (batch_size x output_size)
    CUBLAS_CHECK(cublasSgemm(nn->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            OUTPUT_SIZE, batch_size, HIDDEN_SIZE,
                            &alpha,
                            nn->weights2, HIDDEN_SIZE,   // weights2 = (output_size x hidden_size), transpose -> (hidden_size x output_size)
                            d_hidden, HIDDEN_SIZE,       // hidden = (batch_size x hidden_size)
                            &beta,
                            d_output, OUTPUT_SIZE));     // output = (batch_size x output_size)

    bias_add_kernel<<<(batch_size * OUTPUT_SIZE + 255) / 256, 256>>>(d_output, nn->bias2, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Softmax
    softmax_kernel<<<batch_size, 1>>>(d_output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void save_hidden_values(float *d_hidden_original, float *d_hidden, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_hidden_original[idx] = d_hidden[idx];
    }
}

__global__ void apply_relu_derivative(float *d_grad_hidden, float *d_hidden_original, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_grad_hidden[idx] *= (d_hidden_original[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// Use cuBLAS -> back propagation
void backward_cublas(CuBLASNeuralNetwork *nn, float *d_input, float *d_hidden, float *d_output, int *d_labels,
                     float *d_grad_output, float *d_grad_hidden, float *d_hidden_original, int batch_size) {
    const float alpha = 1.0f, beta = 0.0f;

    save_hidden_values<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden_original, d_hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    dim3 block_size(256);
    dim3 grid_size((OUTPUT_SIZE + block_size.x - 1) / block_size.x, batch_size);
    compute_output_gradient<<<grid_size, block_size>>>(d_output, d_labels, d_grad_output, OUTPUT_SIZE, batch_size);
    CUDA_CHECK(cudaGetLastError());

    // Grad of layer 2: 计算第二层权重梯度：grad_weights2 = d_hidden.T @ d_grad_output
    CUBLAS_CHECK(cublasSgemm(nn->handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            HIDDEN_SIZE, OUTPUT_SIZE, batch_size,
                            &alpha,
                            d_hidden, HIDDEN_SIZE,
                            d_grad_output, OUTPUT_SIZE,
                            &beta,
                            nn->grad_weights2, HIDDEN_SIZE));

    compute_bias_gradient<<<(OUTPUT_SIZE + 255) / 256, 256>>>(nn->grad_bias2, d_grad_output, OUTPUT_SIZE, batch_size);
    CUDA_CHECK(cudaGetLastError());

    // Grad of hidden layer: d_grad_hidden = d_grad_output @ weights2
    CUBLAS_CHECK(cublasSgemm(nn->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            HIDDEN_SIZE, batch_size, OUTPUT_SIZE,
                            &alpha,
                            nn->weights2, HIDDEN_SIZE,
                            d_grad_output, OUTPUT_SIZE,
                            &beta,
                            d_grad_hidden, HIDDEN_SIZE));

    apply_relu_derivative<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_grad_hidden, d_hidden_original, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Grad of layer 1: grad_weights1 = d_input.T @ d_grad_hidden
    CUBLAS_CHECK(cublasSgemm(nn->handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            INPUT_SIZE, HIDDEN_SIZE, batch_size,
                            &alpha,
                            d_input, INPUT_SIZE,
                            d_grad_hidden, HIDDEN_SIZE,
                            &beta,
                            nn->grad_weights1, INPUT_SIZE));

    compute_bias_gradient<<<(HIDDEN_SIZE + 255) / 256, 256>>>(nn->grad_bias1, d_grad_hidden, HIDDEN_SIZE, batch_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

void update_weights(CuBLASNeuralNetwork *nn) {
    int block_size = 256;
    int grid_size;

    grid_size = (HIDDEN_SIZE * INPUT_SIZE + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(nn->weights1, nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE, LEARNING_RATE);

    grid_size = (OUTPUT_SIZE * HIDDEN_SIZE + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(nn->weights2, nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE, LEARNING_RATE);

    grid_size = (HIDDEN_SIZE + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(nn->bias1, nn->grad_bias1, HIDDEN_SIZE, LEARNING_RATE);

    grid_size = (OUTPUT_SIZE + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(nn->bias2, nn->grad_bias2, OUTPUT_SIZE, LEARNING_RATE);

    CUDA_CHECK(cudaDeviceSynchronize());
}

float cross_entropy_loss(float *output, int *labels, int batch_size) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        total_loss -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7f));
    }
    return total_loss / batch_size;
}

float evaluate_accuracy(CuBLASNeuralNetwork *nn, float *d_X_test, int *d_y_test, 
                       float *d_hidden, float *d_output, int total_size) {
    int num_batches = (total_size + BATCH_SIZE - 1) / BATCH_SIZE;
    int total_correct = 0;
    int total_processed = 0;

    for (int batch = 0; batch < num_batches; batch++) {
        int current_batch_size = (batch == num_batches - 1) ? 
            (total_size - batch * BATCH_SIZE) : BATCH_SIZE;
        
        if (current_batch_size <= 0) break;

        forward_cublas(nn, &d_X_test[batch * BATCH_SIZE * INPUT_SIZE], 
                      d_hidden, d_output, current_batch_size);
        
        float *h_output = (float *)malloc(current_batch_size * OUTPUT_SIZE * sizeof(float));
        int *h_y_test = (int *)malloc(current_batch_size * sizeof(int));
        
        CUDA_CHECK(cudaMemcpy(h_output, d_output, 
            current_batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y_test, &d_y_test[batch * BATCH_SIZE], 
            current_batch_size * sizeof(int), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < current_batch_size; i++) {
            int predicted = 0;
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (h_output[i * OUTPUT_SIZE + j] > h_output[i * OUTPUT_SIZE + predicted]) {
                    predicted = j;
                }
            }
            if (predicted == h_y_test[i]) {
                total_correct++;
            }
        }
        total_processed += current_batch_size;
        free(h_output);
        free(h_y_test);
    }
    return 100.0f * total_correct / total_processed;
}

void train(CuBLASNeuralNetwork *nn, float *X_train, int *y_train, float *X_test, int *y_test) {
    float *d_X_train, *d_X_test, *d_hidden, *d_output;
    int *d_y_train, *d_y_test;
    float *d_grad_output, *d_grad_hidden, *d_hidden_original;

    CUDA_CHECK(cudaMalloc(&d_X_train, TRAIN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X_test, TEST_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_train, TRAIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y_test, TEST_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grad_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden_original, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X_train, X_train, TRAIN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X_test, X_test, TEST_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_train, y_train, TRAIN_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_test, y_test, TEST_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    int num_batches = TRAIN_SIZE / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;

        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE;
            
            forward_cublas(nn, &d_X_train[start_idx * INPUT_SIZE], d_hidden, d_output, BATCH_SIZE);

            float *h_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
            CUDA_CHECK(cudaMemcpy(h_output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            float loss = cross_entropy_loss(h_output, &y_train[start_idx], BATCH_SIZE);
            total_loss += loss;
            free(h_output);

            backward_cublas(nn, &d_X_train[start_idx * INPUT_SIZE], d_hidden, d_output, &d_y_train[start_idx],
                           d_grad_output, d_grad_hidden, d_hidden_original, BATCH_SIZE);
            
            update_weights(nn);

            if ((batch + 1) % 100 == 0 || (epoch == 0 && batch == 0)) {
                int test_start_idx = rand() % (TEST_SIZE - BATCH_SIZE);
                float test_accuracy = evaluate_accuracy(nn, 
                    &d_X_test[test_start_idx * INPUT_SIZE],
                    &d_y_test[test_start_idx],
                    d_hidden, d_output, BATCH_SIZE);
                
                printf("Epoch %d/%d, Iter %d/%d, Loss: %.4f, Test Accuracy: %.2f%%\n", 
                       epoch + 1, EPOCHS, batch + 1, num_batches, 
                       total_loss / (batch + 1), test_accuracy);
            }
        }
        
        float test_accuracy = evaluate_accuracy(nn, d_X_test, d_y_test, d_hidden, d_output, TEST_SIZE);
        printf("Epoch %d/%d completed, Loss: %.4f, Test Accuracy: %.2f%%\n", 
            epoch + 1, EPOCHS, total_loss / num_batches, test_accuracy);
    }
    
    CUDA_CHECK(cudaFree(d_X_train));
    CUDA_CHECK(cudaFree(d_X_test));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_y_train));
    CUDA_CHECK(cudaFree(d_y_test));
    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_grad_hidden));
    CUDA_CHECK(cudaFree(d_hidden_original));
}

void cleanup_neural_network(CuBLASNeuralNetwork *nn) {
    CUDA_CHECK(cudaFree(nn->weights1));
    CUDA_CHECK(cudaFree(nn->weights2));
    CUDA_CHECK(cudaFree(nn->bias1));
    CUDA_CHECK(cudaFree(nn->bias2));
    CUDA_CHECK(cudaFree(nn->grad_weights1));
    CUDA_CHECK(cudaFree(nn->grad_weights2));
    CUDA_CHECK(cudaFree(nn->grad_bias1));
    CUDA_CHECK(cudaFree(nn->grad_bias2));
    CUBLAS_CHECK(cublasDestroy(nn->handle));
}

int main() {
    srand(time(NULL));

    printf("=== cuBLAS Neural Network Configuration ===\n");
    printf("      Input Size:        %d\n", INPUT_SIZE);
    printf("      Hidden Size:       %d\n", HIDDEN_SIZE);
    printf("      Output Size:       %d\n", OUTPUT_SIZE);
    printf("      Training Set Size: %d\n", TRAIN_SIZE);
    printf("      Testing Set Size:  %d\n", TEST_SIZE);
    printf("      Batch Size:        %d\n", BATCH_SIZE);
    printf("      Epochs:            %d\n", EPOCHS);
    printf("      Learning Rate:     %.3f\n", LEARNING_RATE);
    printf("==========================================\n");

    CuBLASNeuralNetwork nn;
    initialize_neural_network(&nn);

    float *X_train = (float *)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = (int *)malloc(TRAIN_SIZE * sizeof(int));
    float *X_test = (float *)malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = (int *)malloc(TEST_SIZE * sizeof(int));

    load_data("../data/processed/train_images.bin", X_train, TRAIN_SIZE * INPUT_SIZE);
    load_labels("../data/processed/train_labels.bin", y_train, TRAIN_SIZE);
    load_data("../data/processed/test_images.bin", X_test, TEST_SIZE * INPUT_SIZE);
    load_labels("../data/processed/test_labels.bin", y_test, TEST_SIZE);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    train(&nn, X_train, y_train, X_test, y_test);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double training_time = (end.tv_sec - start.tv_sec) + 
                          (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("\nTotal training time: %.2f sec\n", training_time);

    cleanup_neural_network(&nn);
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}