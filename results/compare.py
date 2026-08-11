import re
import matplotlib.pyplot as plt

def parse_log_from_text(text):
    epochs, loss, acc, latency = [], [], [], []

    for line in text.split("\n"):
        if "Epoch" in line:
            e = int(re.search(r"Epoch (\d+)", line).group(1))
            l = float(re.search(r"Loss: ([0-9.]+)", line).group(1))
            a = float(re.search(r"Train accuracy: ([0-9.]+)", line).group(1))
            epochs.append(e)
            loss.append(l)
            acc.append(a)

        if "Latency" in line:
            val = float(re.search(r"Latency: ([0-9.]+)", line).group(1))
            latency.append(val)

    return epochs, loss, acc, latency


with open("mobile_output.txt") as f:
    mobile_text = f.read()

with open("resnet_output.txt") as f:
    resnet_text = f.read()


mobile_epochs, mobile_loss, mobile_acc, mobile_latency = parse_log_from_text(mobile_text)
res_epochs, res_loss, res_acc, res_latency = parse_log_from_text(resnet_text)


plt.figure()
plt.plot(res_epochs, res_acc, label="ResNet18")
plt.plot(mobile_epochs, mobile_acc, label="MobileNet")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig("Accuracy.png")
plt.show()

plt.figure()
plt.plot(res_epochs, res_loss, label="ResNet18")
plt.plot(mobile_epochs, mobile_loss, label="MobileNet")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Loss.png")
plt.show()

plt.figure()
plt.hist(res_latency, bins=20, alpha=0.5, label="ResNet18")
plt.hist(mobile_latency, bins=20, alpha=0.5, label="MobileNet")
plt.title("Latency Distribution")
plt.xlabel("Seconds")
plt.ylabel("Count")
plt.legend()
plt.savefig("Latency.png")
plt.show()

print("MobileNet avg latency:", sum(mobile_latency)/len(mobile_latency))
print("ResNet avg latency:", sum(res_latency)/len(res_latency))
print("MobileNet final acc:", mobile_acc[-1])
print("ResNet final acc:", res_acc[-1])