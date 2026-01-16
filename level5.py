# LEVEL 4 – ENSEMBLE

print("\n--- STARTING LEVEL 4: ENSEMBLE ---")

print("Loading Level 2 Model...")
model_A = torchvision.models.resnet18(pretrained=False)
model_A.fc = nn.Linear(512, 102)
model_A.load_state_dict(torch.load("level2_model.pth"))
model_A = model_A.to(device)
model_A.eval()

print("Loading Level 3 Model...")
model_B = MyCustomModel()
model_B.load_state_dict(torch.load("level3_custom_model.pth"))
model_B = model_B.to(device)
model_B.eval()

print("Running prediction on Test Set...")
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        out_A = model_A(images)
        out_B = model_B(images)

        final_out = (out_A + out_B) / 2
        _, predicted = torch.max(final_out, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_acc = 100 * correct / total

print("\n*********************************************")
print(f"FINAL ENSEMBLE ACCURACY: {final_acc:.2f}%")
print("*********************************************")

if final_acc > 90:
    print("Great result! This should pass the shortlist criteria.")
else:
    print("Good result, but consider training longer.")
