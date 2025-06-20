import csv
import os
import pandas as pd
from datetime import datetime

FILE = 'expenses.csv'

def init_file():
    if not os.path.exists(FILE):
        with open(FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Category', 'Description', 'Amount'])

def add_expense():
    date = input("Enter date (YYYY-MM-DD): ") or datetime.today().strftime('%Y-%m-%d')
    category = input("Enter category (Food, Travel, etc.): ")
    desc = input("Enter description: ")
    amount = input("Enter amount: ")

    with open(FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([date, category, desc, amount])
    print("✅ Expense added.")

def view_expenses():
    df = pd.read_csv(FILE)
    print(df)

def update_expense():
    df = pd.read_csv(FILE)
    print(df)
    index = int(input("Enter index of row to update: "))
    if 0 <= index < len(df):
        df.loc[index, 'Date'] = input("New date (YYYY-MM-DD): ") or df.loc[index, 'Date']
        df.loc[index, 'Category'] = input("New category: ") or df.loc[index, 'Category']
        df.loc[index, 'Description'] = input("New description: ") or df.loc[index, 'Description']
        df.loc[index, 'Amount'] = input("New amount: ") or df.loc[index, 'Amount']
        df.to_csv(FILE, index=False)
        print("✅ Expense updated.")
    else:
        print("❌ Invalid index.")

def delete_expense():
    df = pd.read_csv(FILE)
    print(df)
    index = int(input("Enter index of row to delete: "))
    if 0 <= index < len(df):
        df = df.drop(index)
        df.to_csv(FILE, index=False)
        print("✅ Expense deleted.")
    else:
        print("❌ Invalid index.")

def show_total():
    df = pd.read_csv(FILE)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    print(f"💰 Total Expenses: ₹{df['Amount'].sum():.2f}")

def menu():
    while True:
        print("\n1. Add Expense\n2. View Expenses\n3. Update Expense\n4. Delete Expense\n5. Show Total\n6. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            add_expense()
        elif choice == '2':
            view_expenses()
        elif choice == '3':
            update_expense()
        elif choice == '4':
            delete_expense()
        elif choice == '5':
            show_total()
        elif choice == '6':
            print("👋 Exiting. Thank you!")
            break
        else:
            print("❌ Invalid choice.")

if __name__ == "__main__":
    init_file()
    menu()
