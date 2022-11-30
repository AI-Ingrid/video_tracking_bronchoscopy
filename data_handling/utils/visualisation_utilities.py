import matplotlib.pyplot as plt


def show_frame_from_batch(sample_from_batch):
    """Show image for a batch of samples."""
    frame_batch, label_batch = sample_from_batch['frame'], sample_from_batch['label']
    #print(frame_batch)
    plt.imshow(frame_batch.permute(1, 2, 0))

