import matplotlib.pyplot as plt
import os

from pandas import DataFrame, Series


class Visualizer:

    @staticmethod
    def plot_comparison(df, col, title, x_label, y_label):
        # Group the data by the 'hasEmoji' column
        grouped = df.groupby(col).size()

        # Create a new figure object for the plot
        fig = plt.figure()

        # Create the plot
        ax = grouped.plot(kind='bar')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        # Set the rotation of the x-axis tick labels
        plt.xticks(rotation=0)

        # Add text labels to the bars
        for i, v in enumerate(grouped.values):
            ax.text(i, v + 0.2, str(v), ha='center')

        fig.set_size_inches(6, 4)

        # Save the plot to a file in the 'images' directory
        if not os.path.exists('images'):
            os.makedirs('images')
        plt.savefig(f'images/{title}.png')

    @staticmethod
    def plot_emoji_emoticon_comparison(set_name, with_emoji, without_emoji):

        # Create a new figure object for the plot
        fig, ax = plt.subplots()

        # Create the plot
        ax.bar(['With', 'Without'], [with_emoji, without_emoji])
        ax.set_xlabel('Emoji/Emoticon Presence')
        ax.set_ylabel('Count')
        ax.set_title(f'{set_name}\nNumber of Tweets: {with_emoji + without_emoji}')

        # Add text labels to the plot
        height = max(with_emoji, without_emoji)
        for i, v in enumerate([with_emoji, without_emoji]):
            ax.text(i, v + height * 0.01, str(v), ha='center')

        # Set the size of the plot
        fig.set_size_inches(4, 4)

        # Save the plot to a file in the 'images' directory
        if not os.path.exists('images'):
            os.makedirs('images')
        plt.savefig(f'images/{set_name}.png', dpi=300)
