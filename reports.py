from fpdf import FPDF
import pandas as pd
import re

def generate_pdf(report_name, data, filename='report.pdf'):
    # Init
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Set basic fields
    pdf.set_title(report_name)
    pdf.set_author("Brandon Galloway")
    pdf.set_font("Times", 'B', 16)
    pdf.cell(0, 10, report_name, 0, 1, 'C')

    pdf.ln(10)

    # Recursive Section splitting based on dict levels
    def add_section(pdf, data, level=0):
        if level == 0:
            pdf.set_font("Arial", size=12)
        else:
            pdf.set_font("Arial", 'B', 14 - level)

        for key, value in data.items():
            if isinstance(value, dict):
                pdf.ln(5)
                pdf.cell(0, 10, f"{key}:", 0, 1)
                add_section(pdf, value, level + 1)
            else:
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, f"{key}: {value}", 0, 1)

    add_section(pdf, data)

    pdf.output("data/clean/reports/"+filename)


def generate_data_quality_report(menus, menu_pages, menu_items, dishes, fileName):
    def check_relationship_constraint(df_left, df_right, left_col, right_col):
        valid_ids = set(df_right[right_col].tolist())
        df_filtered = df_left[df_left[left_col].isin(valid_ids)]
        count = df_left.shape[0] - df_filtered.shape[0]
        return count

    def get_special_character_records(df, col):
        special_characters = r'?[]\.&,%"/()-'
        special_char_pattern = f'[{re.escape(special_characters)}]'
        filtered_records = df[df[col].notna()].copy()
        special_char_records = filtered_records[col].str.contains(special_char_pattern, regex=True)
        filtered_records['contains_special_chars'] = special_char_records
        return filtered_records[filtered_records['contains_special_chars']]

    def get_incorrect_menu_page_count(menus, menu_pages):
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
        return result[result['page_count'] != result['menu_page_id']].shape[0]

    def get_incorrect_menu_dish_counts(menus, menu_pages, menu_items):
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
        return result[result['dish_count'] != result['unique_dish_count']].shape[0]

    def get_incorrect_dish_appearances(menus, menu_pages, menu_items, dishes):
        # Join to Items
        result = pd.merge(dishes[['id']], menu_items[['id', 'menu_page_id', 'dish_id']], left_on='id',
                          right_on='dish_id',
                          how='inner')
        result = result.rename(columns={'id_y': 'menu_item_id', 'id_x': 'id'})

        # Join to Pages
        result = pd.merge(result[['id', 'menu_page_id']], menu_pages[['id', 'menu_id']], left_on='menu_page_id',
                          right_on='id', how='inner')
        result = result.rename(columns={'id_x': 'id'})

        # Join to Menus
        result = pd.merge(result[['id', 'menu_id']], menus[['id', 'date']], left_on='menu_id', right_on='id',
                          how='inner')
        result = result.rename(columns={'id_x': 'id'})
        result = result[['id', 'date']]
        result['date'] = pd.to_datetime(result['date'], errors='coerce')
        # Calculate unique dish counts
        result = result.groupby('id')['date'].agg(['max', 'min']).reset_index()
        result = result.rename(columns={
            'max': 'date_max',
            'min': 'date_min'
        })

        # Merge with menus to update dish counts
        result = dishes.copy().merge(result, left_on='id', right_on='id', how='left')
        return result[result['first_appeared'] != result['date_min']].shape[0], result[result['last_appeared'] != result['date_max']].shape[0]

    def get_incorrect_dish_prices(menu_items, dishes):
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
        return result[result['highest_price'] != result['max_price']].shape[0], result[result['lowest_price'] != result['min_price']].shape[0]

    def get_incorrect_dish_menu_appearances(menus, menu_pages, menu_items, dishes):
        # Join to Items
        result = pd.merge(dishes[['id']], menu_items[['id', 'menu_page_id', 'dish_id']], left_on='id',
                          right_on='dish_id',
                          how='inner')
        result = result.rename(columns={'id_y': 'menu_item_id', 'id_x': 'id'})

        # Join to Pages
        result = pd.merge(result[['id', 'menu_page_id']], menu_pages[['id', 'menu_id']], left_on='menu_page_id',
                          right_on='id', how='inner')
        result = result.rename(columns={'id_x': 'id'})

        # Join to Menus
        result = pd.merge(result[['id', 'menu_id']], menus[['id', 'date']], left_on='menu_id', right_on='id',
                          how='inner')
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
        return result[result['menus_appeared'] != result['count_distinct']].shape[0], result[result['times_appeared'] != result['count']].shape[0]

    data_quality_data = {
        'Orphan Data': {
            'Orphan Record Count': 0
        },
        'Null Columns': {},
        'Special Characters': {},
        'Odd Data': {},
        'Recalculated Fields': {},
    }

    # Add Orphan Section
    removed = check_relationship_constraint(menus.copy(), menu_pages.copy(), 'id', 'menu_id')
    data_quality_data['Orphan Data']['Orphan Record Count'] += removed
    data_quality_data['Orphan Data']['Orphan Record Count (Menus without Pages)'] = removed

    removed = check_relationship_constraint(menu_pages.copy(), menus.copy(), 'menu_id', 'id')
    data_quality_data['Orphan Data']['Orphan Record Count'] += removed
    data_quality_data['Orphan Data']['Orphan Record Count (Pages without Menus)'] = removed

    removed = check_relationship_constraint(menu_pages.copy(), menu_items.copy(), 'id', 'menu_page_id')
    data_quality_data['Orphan Data']['Orphan Record Count'] += removed
    data_quality_data['Orphan Data']['Orphan Record Count (Pages without Items)'] = removed

    removed = check_relationship_constraint(menu_items.copy(), menu_pages.copy(), 'menu_page_id', 'id')
    data_quality_data['Orphan Data']['Orphan Record Count'] += removed
    data_quality_data['Orphan Data']['Orphan Record Count (Items without Pages)'] = removed

    removed = check_relationship_constraint(menu_items.copy(), dishes.copy(), 'dish_id', 'id')
    data_quality_data['Orphan Data']['Orphan Record Count'] += removed
    data_quality_data['Orphan Data']['Orphan Record Count (Items without Dishes)'] = removed

    removed = check_relationship_constraint(dishes.copy(), menu_items.copy(), 'id', 'dish_id')
    data_quality_data['Orphan Data']['Orphan Record Count'] += removed
    data_quality_data['Orphan Data']['Orphan Record Count (Dishes without Items)'] = removed

    # Orphan Counts
    data_quality_data['Null Columns']['Null Column Count (Menu)'] = int(menus.isna().all().sum())
    data_quality_data['Null Columns']['Null Column Count (MenuPage)'] = int(menu_pages.isna().all().sum())
    data_quality_data['Null Columns']['Null Column Count (MenuItem)'] = int(menu_items.isna().all().sum())
    data_quality_data['Null Columns']['Null Column Count (Dish)'] = int(dishes.isna().all().sum())

    # Special Character Issues
    data_quality_data['Special Characters']['Effected Records (Menu: currency)'] = get_special_character_records(menus, 'currency').shape[0]
    data_quality_data['Special Characters']['Effected Records (Menu: currency_symbol)'] = get_special_character_records(menus, 'currency_symbol').shape[0]
    data_quality_data['Special Characters']['Effected Records (Dish: name)'] = get_special_character_records(dishes, 'name').shape[0]

    # Odd Data
    data_quality_data['Odd Data']['Null Priced Menu Items'] = menu_items[menu_items['price'].isna()].shape[0]
    data_quality_data['Odd Data']['Null Dated Menus'] = menus[menus['date'].isna()].shape[0]
    data_quality_data['Odd Data']['Menu Item Price Outliers'] = menu_items[(menu_items['price'] <= 0.0) | (menu_items['price'] > 1500)].shape[0]

    # Recalculated Fields
    data_quality_data['Recalculated Fields']['Incorrect Menu Page Counts'] = get_incorrect_menu_page_count(menus.copy(), menu_pages.copy())
    data_quality_data['Recalculated Fields']['Incorrect Menu Dish Counts'] = get_incorrect_menu_dish_counts(menus, menu_pages, menu_items)
    data_quality_data['Recalculated Fields']['Incorrect Dish First Appeared'], data_quality_data['Incorrect Dish Last Appeared'] = get_incorrect_dish_appearances(menus, menu_pages, menu_items, dishes)
    data_quality_data['Recalculated Fields']['Incorrect Dish Low Price'], data_quality_data['Incorrect Dish High Price'] = get_incorrect_dish_prices(menu_items, dishes)

    # XPOS/YPOS Data
    pd.options.display.float_format = '{:.3f}'.format
    data_quality_data['XPOS/YPOS Data'] = menu_items[['xpos', 'ypos']].describe().astype(float).to_dict()

    menu_items['updated_at'] = pd.to_datetime(menu_items['updated_at'], errors='coerce')
    menu_items['created_at'] = pd.to_datetime(menu_items['created_at'], errors='coerce')
    data_quality_data['Invalid Menu Item Updated At Values'] = menu_items[menu_items['updated_at'] < menu_items['created_at']].shape[0]

    data_quality_data['Incorrect Dish First Appearance'], data_quality_data['Incorrect Dish Last Appearance'] = get_incorrect_dish_menu_appearances(menus, menu_pages, menu_items, dishes)

    # Generate Report
    generate_pdf("Data Quality Report", data_quality_data, filename=fileName+'.pdf')

