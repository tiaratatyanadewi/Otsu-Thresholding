import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random


def total_pix(image):
    size = image.shape[0] * image.shape[1]
    return size


def get_best_threshold(img_array):
    height, width = img_array.shape
    count_pixel = np.zeros(256)

    for i in range(height):
        for j in range(width):
            count_pixel[int(img_array[i][j])] += 1

    max_variance = 0.0
    best_threshold = 0
    for threshold in range(256):
        n0 = count_pixel[:threshold].sum()
        n1 = count_pixel[threshold:].sum()
        w0 = n0 / (height * width)
        w1 = n1 / (height * width)
        u0 = 0.0
        u1 = 0.0

        if n0 != 0:
            u0 = (count_pixel[:threshold] * np.arange(threshold)).sum() / n0
        if n1 != 0:
            u1 = (count_pixel[threshold:] * np.arange(threshold, 256)).sum() / n1

        tmp_var = w0 * w1 * (u0 - u1) ** 2

        if tmp_var > max_variance:
            best_threshold = threshold
            max_variance = tmp_var

    return best_threshold


def my_otsu(image, threshold):
    image = np.transpose(np.asarray(image))
    total = total_pix(image)
    bin_image = image < threshold
    sumT = np.sum(image)
    w0 = np.sum(bin_image)
    sum0 = np.sum(bin_image * image)
    w1 = total - w0
    if w1 == 0:
        return 0
    sum1 = sumT - sum0
    mean0 = sum0 / w0 if w0 > 0 else 0
    mean1 = sum1 / w1 if w1 > 0 else 0
    varBetween = w0 / total * w1 / total * (mean0 - mean1) ** 2
    return varBetween


def bin_to_oct(chrom):
    return sum(val * (2**idx) for idx, val in enumerate(reversed(chrom)))


def init_chrome(N):
    return [np.random.randint(0, 2, 10).tolist() for _ in range(N)]


def get_fitness(image, population):
    test_nums = [bin_to_oct(chrom) for chrom in population]
    fitness = [my_otsu(image, num) for num in test_nums]
    return fitness


def select(image, population, N):
    fitness = get_fitness(image, population)
    sum_fitness = np.sum(fitness)
    probability = fitness / sum_fitness
    accu_probability = np.cumsum(probability)

    new_population = []
    random_nums = np.random.random(N)
    for num in random_nums:
        for i in range(len(accu_probability)):
            if num <= accu_probability[i]:
                new_population.append(population[i])
                break

    while len(new_population) < N:
        new_population.append(np.random.randint(0, 2, 10).tolist())

    return new_population[:N]


def cross(population):
    num1, num2 = random.sample(range(len(population)), 2)
    cross_bits_num = 4
    population[num1][cross_bits_num:], population[num2][cross_bits_num:] = (
        population[num2][cross_bits_num:],
        population[num1][cross_bits_num:],
    )
    return population


def mutate(population, N):
    mutate_num = int(0.06 * N * 10)
    for _ in range(mutate_num):
        x = random.randint(0, N - 1)
        y = random.randint(0, 9)
        population[x][y] = 1 - population[x][y]  # Flip the bit
    return population


def genetic_algorithm_otsu(image, N, max_iteration):
    population = init_chrome(N)
    best_thresholds = []
    best_fitnesss = []
    best_threshold = 0
    best_fitness = -1
    sustain_num = 0
    cata_count = 0

    for _ in range(max_iteration):
        population = select(image, population, N)
        population = cross(population)
        population = mutate(population, N)

        fitness = get_fitness(image, population)
        max_fitness = np.max(fitness)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_threshold = bin_to_oct(population[np.argmax(fitness)])
            sustain_num = 0
        else:
            sustain_num += 1
            if sustain_num >= 5:
                for i in range(len(population)):
                    if my_otsu(image, bin_to_oct(population[i])) == best_fitness:
                        population[i] = init_chrome(1)[0]
                        cata_count += 1
                        if cata_count == 5:
                            sustain_num = 0
                            cata_count = 0

        best_thresholds.append(best_threshold)
        best_fitnesss.append(best_fitness)
        print(f"Best threshold: {best_threshold}, Best fitness: {best_fitness}")

    return best_threshold, best_fitness, best_thresholds, best_fitnesss


def threshold(t, image):
    image_tmp = np.asarray(image)
    intensity_array = np.where(image_tmp < t, 0, 255).astype(np.uint8).flatten()
    new_image = Image.fromarray(intensity_array.reshape(image_tmp.shape))
    new_image.save("image/output.jpg")
    return new_image


def main():
    im = Image.open("image/kuda.jpg")
    im_gray = im.convert("L")

    best_threshold, best_fitness, thresholds, fitnesss = genetic_algorithm_otsu(
        np.array(im_gray), N=10, max_iteration=100
    )
    threshold_image = threshold(best_threshold, im_gray)

    # Display original and thresholded images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(im_gray, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Thresholded Image (T={best_threshold})")
    plt.imshow(threshold_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Plot the results
    plt.figure()
    plt.title(f"Threshold (N=10 Iteration=100)", fontsize=12)
    plt.plot(thresholds, linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Threshold")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title(f"Fitness (N=10 Iteration=100)", fontsize=12)
    plt.plot(fitnesss, linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
