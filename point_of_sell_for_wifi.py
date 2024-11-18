import random
import string

class WifiCodePOS:
    def __init__(self):
        self.codes = {}
        self.sold_codes = {}

    def generate_wifi_code(self):
        """Generates a unique WiFi code."""
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        while code in self.codes:
            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        self.codes[code] = {"price": 5.0}  # Example price for each code
        return code

    def display_available_codes(self):
        """Displays available WiFi codes."""
        if not self.codes:
            print("No available codes.")
            return
        print("\nAvailable WiFi Codes (for sale):")
        for code, details in self.codes.items():
            print(f"Code: {code}, Price: ${details['price']:.2f}")

    def sell_code(self, code):
        """Sells a WiFi code if available."""
        if code not in self.codes:
            print("Code not available for sale.")
            return
        price = self.codes[code]["price"]
        print(f"Selling WiFi code: {code} for ${price:.2f}")
        self.sold_codes[code] = self.codes.pop(code)
        print("Sale successful!\n")

    def display_sold_codes(self):
        """Displays sold WiFi codes."""
        if not self.sold_codes:
            print("No codes have been sold.")
            return
        print("\nSold WiFi Codes:")
        for code, details in self.sold_codes.items():
            print(f"Code: {code}, Price: ${details['price']:.2f}")

def main():
    pos = WifiCodePOS()
    while True:
        print("\n--- WiFi Code POS System ---")
        print("1. Generate New WiFi Code")
        print("2. Display Available Codes")
        print("3. Sell a WiFi Code")
        print("4. Display Sold Codes")
        print("5. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            code = pos.generate_wifi_code()
            print(f"Generated WiFi Code: {code}")
        elif choice == "2":
            pos.display_available_codes()
        elif choice == "3":
            code = input("Enter the WiFi code to sell: ")
            pos.sell_code(code)
        elif choice == "4":
            pos.display_sold_codes()
        elif choice == "5":
            print("Exiting POS system.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
