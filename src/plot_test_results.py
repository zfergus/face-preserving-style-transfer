"""Plot the average face test results."""
import pathlib
import numpy
import matplotlib.pyplot as plt


def plot_face_results(base_losses, face_losses, title):
    """Plot the face results as averages over 1000 faces."""
    n_results = face_losses.size
    assert n_results == base_losses.size
    average_base_losses = numpy.zeros(int(numpy.ceil(n_results / 1000)))
    average_face_losses = numpy.zeros(int(numpy.ceil(n_results / 1000)))

    # Compute the average for 1000 faces
    for i in range(0, n_results, 1000):
        average_base_losses[i // 1000] = numpy.average(base_losses[i:i+1000])
        average_face_losses[i // 1000] = numpy.average(face_losses[i:i+1000])

    x = numpy.arange(1, 171)
    plt.plot(x, average_base_losses,
             label="average loss without facial preservation")
    plt.plot(x, average_face_losses,
             label="average loss with facial preservation")
    plt.xlabel("1000 faces")
    plt.ylabel("||ψ(yc)-ψ(y)||²")
    plt.title("Average {} Face Loss (Averaged Over 1000 Faces)".format(title))
    plt.legend()
    plt.show()


# Hard coded path to the test results
root = pathlib.Path(__file__).parents[1]
with numpy.load(root / "tests/manga-face-test.npz") as f:
    for k, v in f.items():
        exec("{} = v".format(k))
plot_face_results(base_losses, face_losses, "Manga")
