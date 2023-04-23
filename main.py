import modules_functions

def main():
    df=modules_functions.calc_log_return(modules_functions.load_data_yahoo(tickerSymbol='META'))
    modules_functions.plot_logr_price(df)
    print(modules_functions.descriptive_stats(df))

if __name__ == "__main__":
    main()
