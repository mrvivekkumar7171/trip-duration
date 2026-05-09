import matplotlib.pyplot as plt
import unittest, os

class TestDemo(unittest.TestCase):

    def test_basic_addition(self):
        """A simple test to prove the pipeline works."""
        result = 2 + 2
        expected = 4
        self.assertEqual(result, expected)

    def test_basic_subtraction(self):
        """Another simple test."""
        result = 10 - 5
        expected = 5
        self.assertEqual(result, expected)

    def test_plot_generation(self):
        # --- First Plot: Bar Chart ---
        fig, ax = plt.subplots()

        fruits = ['apple', 'blueberry', 'cherry', 'orange']
        counts = [140, 10, 30, 55]
        bar_labels = ['red', 'blue', '_red', 'orange']
        bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

        ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

        ax.set_ylabel('fruit supply')
        ax.set_title('Fruit supply by kind and color')
        ax.legend(title='Fruit color')

        plt.savefig('bars.png', bbox_inches='tight')
        plt.close(fig) # Good practice to close figures in tests

        # --- Second Plot: Line Chart ---
        cat = ["bored", "happy", "happy", "happy", "happy", "bored"]
        dog = ["bored", "bored", "bored", "happy", "bored", "bored"]
        activity = ["combing", "drinking", "feeding", "napping", "playing", "washing"]

        fig2, ax2 = plt.subplots()
        ax2.plot(activity, dog, label="dog")
        ax2.plot(activity, cat, label="cat")
        ax2.legend()

        plt.savefig('lines.png', bbox_inches='tight')
        print(f"Image saved successfully at: {os.path.abspath('lines.png')}")
        plt.close(fig2)

if __name__ == '__main__':
    unittest.main()