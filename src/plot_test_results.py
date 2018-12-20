"""Plot the average face test results."""
import pathlib
import numpy
import matplotlib.pyplot as plt


def plot_face_results(base_losses, face_losses, title):
    """Plot the face results as averages over 1000 faces."""
    n_results = face_losses.size
    assert n_results == base_losses.size
    # average_base_losses = numpy.zeros(int(numpy.ceil(n_results / 1000)))
    # average_face_losses = numpy.zeros(int(numpy.ceil(n_results / 1000)))

    # Compute the average for 1000 faces
    # for i in range(0, n_results, 1000):
    #     average_base_losses[i // 1000] = numpy.average(base_losses[i:i+1000])
    #     average_face_losses[i // 1000] = numpy.average(face_losses[i:i+1000])

    x = numpy.arange(1, base_losses.size + 1)
    # plt.scatter(x, base_losses, s=1,
    #             label="average loss without facial preservation")
    # plt.plot(x, numpy.full(x.shape, numpy.average(base_losses)),
    #          label="average loss without facial preservation")
    # plt.scatter(x, face_losses, s=1,
    #             label="average loss with facial preservation")
    diff = base_losses - face_losses
    n_good_results = numpy.count_nonzero(diff > 0)
    n_bad_results = face_losses.size - n_good_results
    plt.scatter(x[diff >= 0], diff[diff >= 0], s=1, label=(
        "With facial preservation performed better ({:.3f}%)").format(
        100. * n_good_results / n_results))
    plt.scatter(x[diff < 0], diff[diff < 0], s=1, label=(
        "Without facial preservation performed better ({:.3f}%)").format(
        100. * n_bad_results / n_results))
    avg_diff = numpy.average(diff)
    plt.plot(x, numpy.full(x.shape, avg_diff), c="#d62728",
             label=f"Average difference ({avg_diff:.3g})")
    # plt.plot(x, numpy.full(x.shape, numpy.average(face_losses)),
    #          label="average loss with facial preservation")
    plt.xlabel("Face in VGGFace2 dataset")
    plt.ylabel("Difference in face losses\n(without - with)")
    plt.title("{} With and Without Facial Preservation".format(title))
    plt.legend()
    # plt.ylim((min(base_losses.min(), face_losses.min()) - 1e-5,
    #           max(base_losses.max(), face_losses.max()) + 1e-5))
    plt.ylim((diff.min(), diff.max()))
    plt.xlim((1, base_losses.size + 1))
    plt.show()


# Hard coded path to the test results
root = pathlib.Path(__file__).parents[1]
with numpy.load(root / "tests/manga-face-test.npz") as f:
    for k, v in f.items():
        exec("{} = v - 1".format(k))

plot_face_results(base_losses, face_losses, "Manga")
