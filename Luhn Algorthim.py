from random import choice


def verify_card_number(card_number):
    sum_of_odd_digits = 0
    card_number_reversed = card_number[::-1]
    odd_digits = card_number_reversed[::2]

    for digit in odd_digits:
        sum_of_odd_digits += int(digit)

    sum_of_even_digits = 0
    even_digits = card_number_reversed[1::2]
    for digit in even_digits:
        number = int(digit) * 2
        if number >= 10:
            number = (number // 10) + (number % 10)
        sum_of_even_digits += number
    total = sum_of_odd_digits + sum_of_even_digits

    return total % 10 == 0


def main():
    user_input = input('Do you want to check your account balance? (yes/no): ').lower()
    if user_input == "yes":
        card_number = input('Enter your card number: ')

        # Remove any dashes or spaces from the card number
        card_translation = str.maketrans({'-': '', ' ': ''})
        translated_card_number = card_number.translate(card_translation)

        if verify_card_number(translated_card_number):
            print('VALID!')
        else:
            print('INVALID!')
    else:
        print("Thank you. Have a nice day!")


main()
