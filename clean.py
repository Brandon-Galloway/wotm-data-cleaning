import pandas as pd
import numpy as np
import re
import os
import logging
from functools import wraps
from  reports import generate_data_quality_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# This decorator function is Largely formated and informed through these guides
# https://www.geeksforgeeks.org/decorators-in-python/
# https://www.geeksforgeeks.org/timing-functions-with-decorators-python/
def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"Completed {func.__name__}")
        return result

    return wrapper


# @BEGIN main
# @IN Menu.csv  @URI file:data/Menu.csv
# @IN MenuPage.csv  @URI file:data/MenuPage.csv
# @IN MenuItem.csv  @URI file:data/MenuItem.csv
# @IN Dish.csv  @URI file:data/Dish.csv
# @OUT menus @URI file:data/clean/Menu.csv
# @OUT menu_pages @URI file:data/clean/MenuPage.csv
# @OUT menu_items @URI file:data/clean/MenuItem.csv
# @OUT dishes @URI file:data/clean/Dish.csv
# @OUT menu_descriptions @URI file:data/clean/MenuDescriptions.csv
# @OUT menu_description @URI file:data/clean/MenuDescription.csv
# @OUT starting_data_quality_report @URI file:data/clean/reports/starting_data_quality_report.pdf
# @OUT ending_data_quality_report @URI file:data/clean/reports/ending_data_quality_report.pdf
def main():
    # @BEGIN import_data_files
    # @IN Menu.csv  @URI file:data/Menu.csv
    # @IN MenuPage.csv  @URI file:data/MenuPage.csv
    # @IN MenuItem.csv  @URI file:data/MenuItem.csv
    # @IN Dish.csv  @URI file:data/Dish.csv
    # @OUT menus @AS raw_menu_data
    # @OUT menu_pages @AS raw_menu_page_data
    # @OUT menu_items @AS raw_menu_item_data
    # @OUT dishes @AS raw_dish_data
    menus, menu_pages, menu_items, dishes = import_data_files()
    # @END import_data_files

    # @BEGIN generate_starting_report
    # @IN raw_menu_data
    # @IN raw_menu_page_data
    # @IN raw_menu_item_data
    # @IN raw_dish_data
    # @OUT starting_data_quality_report @URI file:data/clean/reports/starting_data_quality_report.pdf
    generate_data_quality_report(menus, menu_pages, menu_items, dishes,"starting_data_quality_report")
    # @END generate_starting_report

    # @BEGIN remove_orphan_records
    # @IN raw_menu_data
    # @IN raw_menu_page_data
    # @IN raw_menu_item_data
    # @IN raw_dish_data
    # @OUT menus @AS reduced_menu_data
    # @OUT menu_pages @AS reduced_menu_page_data
    # @OUT menu_items @AS reduced_menu_item_data
    # @OUT dishes @AS reduced_dish_data
    menus, menu_pages, menu_items, dishes = remove_orphan_records(menus, menu_pages, menu_items, dishes)
    # @END remove_orphan_records

    # @BEGIN remove_null_or_redundant_fields
    # @IN reduced_menu_data
    # @IN reduced_menu_page_data
    # @IN reduced_dish_data
    # @OUT menus @AS reduced_field_menu_data
    # @OUT menu_pages @AS reduced_field_menu_page_data
    # @OUT dishes @AS reduced_field_dish_data
    menus, menu_pages, dishes = remove_null_or_redundant_fields(menus, menu_pages, dishes)
    # @END remove_null_or_redundant_fields

    # @BEGIN perform_easy_quality_enhancements
    # @IN reduced_field_menu_page_data
    # @OUT menus @AS simply_cleaned_menu_data
    menus = perform_easy_quality_enhancements(menus)
    # @END perform_easy_quality_enhancements

    # @BEGIN clean_special_character_issues
    # @IN simply_cleaned_menu_data
    # @IN reduced_field_dish_data
    # @OUT menus @AS special_character_adjusted_menu_data
    # @OUT dishes @AS special_character_adjusted_dish_data
    menus, dishes = clean_special_character_issues(menus, dishes)
    # @END clean_special_character_issues

    # @BEGIN address_null_values
    # @IN special_character_adjusted_menu_data
    # @IN reduced_menu_item_data
    # @IN special_character_adjusted_dish_data
    # @OUT menu_items @AS denulled_menu_item_data
    menu_items = address_null_values(menus, menu_items, dishes)
    # @END address_null_values

    # @BEGIN address_outlier_values
    # @IN special_character_adjusted_menu_data
    # @IN reduced_field_menu_page_data
    # @IN denulled_menu_item_data
    # @IN special_character_adjusted_dish_data
    # @OUT menus @AS outlier_addressed_menu_data
    # @OUT menu_items @AS outlier_addressed_menu_item_data
    menus, menu_items = address_outlier_values(menus, menu_pages, menu_items, dishes)
    # @END address_outlier_values

    # Repeat
    # @BEGIN remove_orphan_records
    # @IN fk_updated_menu_data
    # @IN reduced_field_menu_page_data
    # @IN outlier_addressed_menu_item_data
    # @IN special_character_adjusted_dish_data
    # @OUT menus @AS re_reduced_menu_data
    # @OUT menu_pages @AS re_reduced_menu_page_data
    # @OUT menu_items @AS re_reduced_menu_item_data
    # @OUT dishes @AS re_reduced_dish_data
    menus, menu_pages, menu_items, dishes = remove_orphan_records(menus, menu_pages, menu_items, dishes)
    # @END remove_orphan_records

    # @BEGIN recalculate_fields
    # @IN re_reduced_menu_data
    # @IN re_reduced_menu_page_data
    # @IN re_reduced_menu_item_data
    # @IN re_reduced_dish_data
    # @OUT menus @AS recalculated_menu_data
    # @OUT dishes @AS recalculated_dish_data
    menus, dishes = recalculate_fields(menus, menu_pages, menu_items, dishes)
    # @END recalculate_fields

    # @BEGIN verify_core_fields
    # @IN re_reduced_menu_item_data
    verify_core_fields(menu_items)
    # @END verify_core_fields

    # @BEGIN schema_changes_for_physical_description
    # @IN recalculated_menu_data
    # @OUT menus @AS fk_updated_menu_data
    # @OUT menu_description @AS menu_description_data
    # @OUT menu_descriptions @AS menu_descriptions_data
    menus, menu_description, menu_descriptions = schema_changes_for_physical_description(menus)
    # @END schema_changes_for_physical_description

    # @BEGIN calculate_dish_menu_appearances
    # @IN fk_updated_menu_data
    # @IN re_reduced_menu_page_data
    # @IN re_reduced_menu_item_data
    # @IN recalculated_dish_data
    # @OUT dishes @AS appearance_recalculated_dish_data
    dishes = calculate_dish_menu_appearances(menus, menu_pages, menu_items, dishes)
    # @END calculate_dish_menu_appearances

    # @BEGIN generate_ending_report
    # @IN fk_updated_menu_data
    # @IN re_reduced_menu_page_data
    # @IN re_reduced_menu_item_data
    # @IN appearance_recalculated_dish_data
    # @OUT ending_data_quality_report @URI file:data/clean/reports/ending_data_quality_report.pdf
    generate_data_quality_report(menus, menu_pages, menu_items, dishes, "ending_data_quality_report")
    # @END generate_ending_report

    # Export
    # @BEGIN save_cleaned_dataframes
    # @IN fk_updated_menu_data
    # @IN re_reduced_menu_page_data
    # @IN re_reduced_menu_item_data
    # @IN appearance_recalculated_dish_data
    # @IN menu_descriptions_data
    # @IN menu_description_data
    # @OUT menus @URI file:data/clean/Menu.csv
    # @OUT menu_pages @URI file:data/clean/MenuPage.csv
    # @OUT menu_items @URI file:data/clean/MenuItem.csv
    # @OUT dishes @URI file:data/clean/Dish.csv
    # @OUT menu_descriptions @URI file:data/clean/MenuDescriptions.csv
    # @OUT menu_description @URI file:data/clean/MenuDescription.csv
    save_cleaned_dataframes(menus, menu_pages, menu_items, dishes, menu_descriptions, menu_description)
    # @END save_cleaned_dataframes


# @END main


@log_execution
def import_data_files():
    menus = pd.read_csv('data/Menu.csv')
    menu_pages = pd.read_csv('data/MenuPage.csv')
    menu_items = pd.read_csv('data/MenuItem.csv')
    dishes = pd.read_csv('data/Dish.csv')
    return menus, menu_pages, menu_items, dishes


@log_execution
def remove_orphan_records(menus, menu_pages, menu_items, dishes):
    def enforce_relationship_constraint(df_left, df_right, left_col, right_col):
        valid_ids = set(df_right[right_col].tolist())
        df_filtered = df_left[df_left[left_col].isin(valid_ids)]
        count = df_left.shape[0] - df_filtered.shape[0]
        print(f'Removed {count} records')
        return df_filtered, count

    total_deleted = 0
    menus, removed = enforce_relationship_constraint(menus, menu_pages, 'id', 'menu_id')
    total_deleted += removed

    menu_pages, removed = enforce_relationship_constraint(menu_pages, menus, 'menu_id', 'id')
    total_deleted += removed

    menu_pages, removed = enforce_relationship_constraint(menu_pages, menu_items, 'id', 'menu_page_id')
    total_deleted += removed

    menu_items, removed = enforce_relationship_constraint(menu_items, menu_pages, 'menu_page_id', 'id')
    total_deleted += removed

    menu_items, removed = enforce_relationship_constraint(menu_items, dishes, 'dish_id', 'id')
    total_deleted += removed

    dishes, removed = enforce_relationship_constraint(dishes, menu_items, 'id', 'dish_id')
    total_deleted += removed

    if total_deleted > 0:
        return remove_orphan_records(menus, menu_pages, menu_items, dishes)

    return menus, menu_pages, menu_items, dishes


@log_execution
def remove_null_or_redundant_fields(menus, menu_pages, dishes):
    menus = menus.drop(columns=['keywords', 'language', 'location_type'])
    menu_pages = menu_pages.drop(columns=['uuid'])
    dishes = dishes.drop(columns=['description'])
    return menus, menu_pages, dishes


# Perform Easy Data Quality Enhancements
# Combine name data under sponsor
@log_execution
def perform_easy_quality_enhancements(menus):
    menus['sponsor'] = menus['sponsor'].fillna(menus['name'])
    menus = menus.drop(columns=['name'])
    return menus


# Clean Special Character Issues
# MENU: CURRENCY CURRENCY_SYMBOL
# DISH: NAME
@log_execution
def clean_special_character_issues(menus, dishes):
    # Remove groupings such as [foo] (foo) "foo"
    # strip_inclusions also removes the wrapped content
    def remove_groups(df, col, strip_inclusions=True):
        if strip_inclusions:
            # Match a little whitespace then (CONTENT) remove all
            df[col] = df[col].str.replace(r'\s*[\(\[\{"\'].*?[\)\]\}"\']', '', regex=True).str.strip()
        else:
            # Match a little whitespace then (CONTENT) remove just the outside
            df[col] = df[col].str.replace(r'[\(\[\{"\'](.*?)[\)\]\}"\']', r'\1', regex=True).str.strip()
        return df

    #
    def remove_unbalanced_groups(df, col):
        def check_balance(s):
            # Stacks for paren/brackets
            stack_paren = []
            stack_bracket = []
            stack_single_quote = []
            stack_double_quote = []
            removals = set()

            # Non-performant: Loop over everything to pop/append
            for i, char in enumerate(s):
                # Handle Paren
                if char == '(':
                    stack_paren.append(i)
                elif char == ')':
                    if stack_paren:
                        stack_paren.pop()
                    else:
                        removals.add(i)
                # Handle Brackets
                elif char == '[':
                    stack_bracket.append(i)
                elif char == ']':
                    if stack_bracket:
                        stack_bracket.pop()
                    else:
                        removals.add(i)
                # Handle Quotes
                elif char == '"':
                    if stack_double_quote:
                        stack_double_quote.pop()
                    else:
                        stack_double_quote.append(i)
                elif char == "'":
                    if stack_single_quote:
                        stack_single_quote.pop()
                    else:
                        stack_single_quote.append(i)

            # Combine all stack findings
            removals.update(stack_paren)
            removals.update(stack_bracket)
            removals.update(stack_double_quote)
            removals.update(stack_single_quote)
            # Apply stack findings
            return ''.join([char for i, char in enumerate(s) if i not in removals])

        # Apply custom function over the column
        df[col] = df[col].apply(check_balance)

        return df

    def remove_special_characters(df, col, special_char_pattern):
        df[col] = df[col].str.replace(special_char_pattern, '', regex=True).str.strip()

    def get_special_character_records(df, col, special_char_pattern):
        filtered_records = df[df[col].notna()].copy()
        special_char_records = filtered_records[col].str.contains(special_char_pattern, regex=True)
        filtered_records['contains_special_chars'] = special_char_records
        return filtered_records[filtered_records['contains_special_chars']]

    def translate_text_counts(df, col):
        digit_mappings = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3',
            'four': '4', 'five': '5', 'six': '6', 'seven': '7',
            'eight': '8', 'nine': '9', 'each': '1'
        }
        pattern = re.compile(r'\((\w+)\)')

        df[col] = df[col].apply(
            lambda x: pattern.sub(
                lambda match: f"({digit_mappings.get(match.group(1).lower(), match.group(1))})",
                x
            ) if pd.notna(x) else x
        )

    def remove_ending_characters(df, col, characters):
        df[col] = df[col].str.replace(f'[{re.escape(characters)}]+$', '', regex=True).str.strip()

    def remove_starting_characters(df, col, characters):
        df[col] = df[col].str.replace(f'^[{re.escape(characters)}]+', '', regex=True).str.strip()
        return df

    # Define regex patterns
    special_characters = r'?[]\.&,%"/()-'
    special_char_pattern = f'[{re.escape(special_characters)}]'

    # MENU - CURRENCY
    remove_groups(menus, 'currency')
    # get_special_character_records(menus,'currency')

    # MENU - CURRENCY_SYMBOL
    remove_special_characters(menus, 'currency_symbol', special_char_pattern)
    # print(set(zip(menus['currency'], menus['currency_symbol'])))

    # DISH - NAME
    translate_text_counts(dishes, 'name')
    remove_groups(dishes, 'name', strip_inclusions=False)
    remove_starting_characters(dishes, 'name', '\\')
    remove_ending_characters(dishes, 'name', '.,-\\')
    remove_unbalanced_groups(dishes, 'name')
    # get_special_character_records(dishes,'name')

    return menus, dishes


def estimate_null_priced_menu_items(menu_items, dishes):
    # Find null price menu items
    null_priced_items = menu_items[menu_items['price'].isna()]

    # Join to Dishes
    result = pd.merge(null_priced_items[['id', 'dish_id']], dishes[['id', 'lowest_price', 'highest_price']],
                      left_on='dish_id', right_on='id', how='left')
    result = result.rename(columns={'id_x': 'id'})

    # Calculate the final result
    result = result[['id', 'lowest_price', 'highest_price']]
    result['price_est'] = ((result['lowest_price'] + result['highest_price']) / 2)
    result = result.groupby('id')['price_est'].median().reset_index()

    result = menu_items.copy().merge(result, left_on='id', right_on='id', how='left')
    result['price'] = result['price_est'].fillna(result['price'])
    result['high_price'] = result['high_price'].fillna(result['price'])
    result = result.drop(columns=['price_est'])
    menu_items = result

    return menu_items


# Address Null Values
# MENU: CURRENCY CURRENCY_SYMBOL
# MENU_ITEMS: PRICE
@log_execution
def address_null_values(menus, menu_items, dishes):
    # MENU - CURRENCY & CURRENCY_SYMBOL
    # Remove records where both are null
    print(menus.shape)
    # Would drop far too many menus :(
    # menus = menus.dropna(subset=['currency', 'currency_symbol'], how='all')
    # print(menus.shape)

    # MENU_ITEMS - PRICE
    menu_items = estimate_null_priced_menu_items(menu_items, dishes)
    menu_items['price'] = menu_items['price'].fillna(menu_items['high_price'])

    return menu_items


def estimate_null_dated_menus(menus, menu_pages, menu_items, dishes):
    # Find null dated menus
    null_dated_menus = menus[menus['date'].isna()]

    # Join to Pages
    result = pd.merge(null_dated_menus[['id']], menu_pages[['id', 'menu_id']], left_on='id', right_on='menu_id',
                      how='inner')
    result = result.rename(columns={'id_y': 'menu_page_id'})

    # Join to Items
    result = pd.merge(result[['menu_id', 'menu_page_id']], menu_items[['id', 'dish_id']], left_on='menu_page_id',
                      right_on='id', how='inner')

    # Join to Dishes
    result = pd.merge(result[['menu_id', 'dish_id']], dishes[['id', 'first_appeared', 'last_appeared']],
                      left_on='dish_id', right_on='id', how='left')

    # Calculate the final result
    result = result[['menu_id', 'first_appeared', 'last_appeared']]
    result['appearance_est'] = ((result['first_appeared'] + result['last_appeared']) // 2).astype(int)
    result = result.groupby('menu_id')['appearance_est'].median().reset_index()
    result['appearance_est'] = result['appearance_est'].astype(int).astype(str) + "-01-01"

    result = menus.copy().merge(result, left_on='id', right_on='menu_id', how='left')
    result['date'] = result['appearance_est'].fillna(result['date'])
    result = result.drop(columns=['menu_id', 'appearance_est'])
    menus = result
    return menus


def clean_dates(date):
    pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    if not isinstance(date, str) or not pattern.match(date):
        return np.nan

    try:
        date = pd.to_datetime(date)
        if date.year < 1840 or date.year > 2024:
            return np.nan
    except ValueError:
        return np.nan

    return date.strftime('%Y-%m-%d')


# Address Outlier Values
# MENU: DATE
# MENU_PAGES: FULL_HEIGHT, FULL_WIDTH
# MENU_ITEMS: PRICE
@log_execution
def address_outlier_values(menus, menu_pages, menu_items, dishes):
    # MENU - DATE
    # Address the phase-1 identified outliers (Query 43)
    # Set them as null for recalculation
    menus['date'] = menus['date'].apply(clean_dates)
    menus = estimate_null_dated_menus(menus, menu_pages, menu_items, dishes)
    menus = menus.dropna(subset=['date'], how='all')
    # print(set(menus['date']))

    # MENU_PAGES - FULL_HEIGHT
    # menu_pages[menu_pages['full_height'].isna()]

    # MENU_PAGES - FULL_WIDTH
    # menu_pages[menu_pages['full_width'].isna()]

    # MENU_ITEMS - PRICE
    # menus.loc[(menus['date'] < '1840-01-01') | (menus['date'] > '2024-12-31')]  = np.nan
    menu_items['price'] = menu_items['price'].astype(float)
    menu_items.drop(menu_items[(menu_items['price'] <= 0.0) | (menu_items['price'] > 1500)].index, inplace=True)

    return menus, menu_items


def calculate_menu_page_counts(menus, menu_pages):
    # Find null dated menus

    # Join to Pages
    result = pd.merge(menus[['id', 'page_count']], menu_pages[['id', 'menu_id']], left_on='id', right_on='menu_id',
                      how='inner')
    result = result.rename(columns={
        'id_y': 'menu_page_id',
        'id_x': 'id'
    })

    # Calculate the final result
    result = result[['id', 'page_count', 'menu_page_id']]
    result = result.groupby('id').agg({'menu_page_id': lambda x: x.nunique()}).reset_index()
    result = menus.copy().merge(result, left_on='id', right_on='id', how='left')
    result['page_count'] = result['menu_page_id']
    result = result.drop(columns=['menu_page_id'])
    menus = result
    return menus


def calculate_menu_dish_counts(menus, menu_pages, menu_items):
    # Join to Pages
    result = pd.merge(menus[['id']], menu_pages[['id', 'menu_id']], left_on='id', right_on='menu_id', how='inner')
    result = result.rename(columns={'id_y': 'menu_page_id', 'id_x': 'id'})

    # Join to Items
    result = pd.merge(result[['id', 'menu_page_id']], menu_items[['id', 'dish_id']], left_on='menu_page_id',
                      right_on='id', how='inner')
    result = result.rename(columns={'id_y': 'menu_item_id', 'id_x': 'id'})

    # Calculate unique dish counts
    unique_dish_counts = result.groupby('id')['dish_id'].nunique().reset_index()
    unique_dish_counts = unique_dish_counts.rename(columns={'dish_id': 'unique_dish_count'})

    # Merge with menus to update dish counts
    result = menus.copy().merge(unique_dish_counts, left_on='id', right_on='id', how='left')
    result['dish_count'] = result['unique_dish_count'].fillna(0).astype(int)
    menus = result.drop(columns=['unique_dish_count'])
    return menus


def calculate_dish_appearances(menus, menu_pages, menu_items, dishes):
    # Join to Items
    result = pd.merge(dishes[['id']], menu_items[['id', 'menu_page_id', 'dish_id']], left_on='id', right_on='dish_id',
                      how='inner')
    result = result.rename(columns={'id_y': 'menu_item_id', 'id_x': 'id'})

    # Join to Pages
    result = pd.merge(result[['id', 'menu_page_id']], menu_pages[['id', 'menu_id']], left_on='menu_page_id',
                      right_on='id', how='inner')
    result = result.rename(columns={'id_x': 'id'})

    # Join to Menus
    result = pd.merge(result[['id', 'menu_id']], menus[['id', 'date']], left_on='menu_id', right_on='id', how='inner')
    result = result.rename(columns={'id_x': 'id'})
    result = result[['id', 'date']]

    # Calculate unique dish counts
    result = result.groupby('id')['date'].agg(['max', 'min']).reset_index()
    result = result.rename(columns={
        'max': 'date_max',
        'min': 'date_min'
    })

    # Merge with menus to update dish counts
    result = dishes.copy().merge(result, left_on='id', right_on='id', how='left')
    result['first_appeared'] = result['date_min']
    result['last_appeared'] = result['date_max']
    dishes = result.drop(columns=['date_max', 'date_min'])
    return dishes


def calculate_dish_prices(menu_items, dishes):
    # Join to Items
    result = pd.merge(dishes[['id']], menu_items[['id', 'menu_page_id', 'dish_id', 'price']], left_on='id',
                      right_on='dish_id', how='inner')
    result = result.rename(columns={'id_y': 'menu_item_id', 'id_x': 'id'})

    # Calculate unique dish counts
    result = result.groupby('id').agg({
        'price': ['max', 'min']
    }).reset_index()
    # Fix odd column names
    result.columns = ['id', 'max_price', 'min_price']

    # Merge with menus to update dish counts
    result = dishes.copy().merge(result, left_on='id', right_on='id', how='left')
    result['highest_price'] = result['max_price']
    result['lowest_price'] = result['min_price']
    dishes = result.drop(columns=['max_price', 'min_price'])
    return dishes


# Recalculate Fields
# MENU: PAGE_COUNT, DISH_COUNT
# MENU_PAGES: PAGE_NUMBER
# MENU_ITEMS:
# DISH: FIRST_APPEARED, LAST_APPEARED, LOWEST_PRICE, HIGHEST_PRICE
@log_execution
def recalculate_fields(menus, menu_pages, menu_items, dishes):
    # MENU - PAGE_COUNT
    menus = calculate_menu_page_counts(menus, menu_pages)

    # MENU - DISH_COUNT
    menus = calculate_menu_dish_counts(menus, menu_pages, menu_items)

    # DISH: FIRST_APPEARED DISH & LAST_APPEARED
    dishes = calculate_dish_appearances(menus, menu_pages, menu_items, dishes)

    # DISH: LOWEST_PRICE & HIGHEST_PRICE
    dishes = calculate_dish_prices(menu_items, dishes)

    return menus, dishes


@log_execution
def verify_core_fields(menu_items):
    # Double Verification of Core Fields
    # MENU_ITEMS: XPOS, YPOS
    pd.options.display.float_format = '{:.3f}'.format
    menu_items[['xpos', 'ypos']].describe().astype(float)


# Schema Changes For Physical Description
@log_execution
def schema_changes_for_physical_description(menus):
    # Create menu_descriptions table
    menu_descriptions = menus[['id', 'physical_description']].copy()
    menu_descriptions['physical_description'] = menu_descriptions['physical_description'].str.split(';')
    menu_descriptions = menu_descriptions.explode('physical_description')

    menu_descriptions['physical_description'] = menu_descriptions['physical_description'].str.strip()
    menu_descriptions = menu_descriptions[menu_descriptions['physical_description'] != '']

    menu_descriptions = menu_descriptions.reset_index(drop=True)

    # Create menu_description table
    menu_description = menu_descriptions['physical_description'].unique()
    menu_description = pd.DataFrame(menu_description, columns=['physical_description'])

    menu_description['id'] = range(1, len(menu_description) + 1)
    menu_description = menu_description[['id', 'physical_description']]
    menu_description.columns = ['id', 'descriptor']

    # Update menu_descriptions to replace the descriptions with a fk to the menu_description table
    menu_descriptions = pd.merge(menu_descriptions, menu_description, left_on='physical_description',
                                 right_on='descriptor', how='left')
    menu_descriptions['physical_description'] = menu_descriptions['id_y']
    menu_descriptions = menu_descriptions[['id_x', 'physical_description']]
    menu_descriptions.columns = ['menu_id', 'physical_desc_id']

    menus = menus.drop(columns=['physical_description'])
    return menus, menu_description, menu_descriptions


# Address Invalid updated_at values
# MENU_ITEM - UPDATED_AT
@log_execution
def address_invalid_updated_at_values(menu_items):
    invalid_stamped_records = menu_items['updated_at'] < menu_items['created_at']
    menu_items.loc[invalid_stamped_records, 'updated_at'] = menu_items.loc[invalid_stamped_records, 'created_at']
    return menu_items


# Recalculate use case adjacent fields
# DISH - MENUS_APPEARED, TIMES_APPEARED
@log_execution
def calculate_dish_menu_appearances(menus, menu_pages, menu_items, dishes):
    # Join to Items
    result = pd.merge(dishes[['id']], menu_items[['id', 'menu_page_id', 'dish_id']], left_on='id', right_on='dish_id',
                      how='inner')
    result = result.rename(columns={'id_y': 'menu_item_id', 'id_x': 'id'})

    # Join to Pages
    result = pd.merge(result[['id', 'menu_page_id']], menu_pages[['id', 'menu_id']], left_on='menu_page_id',
                      right_on='id', how='inner')
    result = result.rename(columns={'id_x': 'id'})

    # Join to Menus
    result = pd.merge(result[['id', 'menu_id']], menus[['id', 'date']], left_on='menu_id', right_on='id', how='inner')
    result = result.rename(columns={
        'id_x': 'id',
    })
    result = result[['id', 'menu_id']]

    # Calculate unique dish counts
    result = result.groupby('id').agg(
        count=('menu_id', 'size'),
        count_distinct=('menu_id', 'nunique')
    ).reset_index()

    # Merge with menus to update dish counts
    result = dishes.copy().merge(result, left_on='id', right_on='id', how='left')
    result['menus_appeared'] = result['count_distinct'].fillna(0).astype(int)
    result['times_appeared'] = result['count'].fillna(0).astype(int)
    dishes = result.drop(columns=['count_distinct', 'count'])
    return dishes


@log_execution
def save_cleaned_dataframes(menus, menu_pages, menu_items, dishes, menu_descriptions, menu_description,
                            save_dir='data/clean'):
    os.makedirs(save_dir, exist_ok=True)
    menus.to_csv(f'{save_dir}/Menu.csv', index=False)
    menu_pages.to_csv(f'{save_dir}/MenuPage.csv', index=False)
    menu_items.to_csv(f'{save_dir}/MenuItem.csv', index=False)
    dishes.to_csv(f'{save_dir}/Dish.csv', index=False)
    menu_descriptions.to_csv(f'{save_dir}/MenuDescriptions.csv', index=False)
    menu_description.to_csv(f'{save_dir}/MenuDescription.csv', index=False)


if __name__ == "__main__":
    main()
