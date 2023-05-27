import utils

def main():
    day_type_user='1d'
    mark_type_user='Close'
    model_name_user='GARCH'
    tickerSymbol='META'
        
    df, tickerSymbol = utils.load_data_yahoo(tickerSymbol, period=day_type_user)
    df=utils.calc_log_return(df,mark_type=mark_type_user)
    n, fcast_dates=utils.pred_interval(data=df,day_type=day_type_user)
    
    # utils.plot_logr_price(data=df,tickerSymbol=tickerSymbol,mark_type=mark_type_user)
    print(utils.descriptive_stats(df))

    #Calling models
    if model_name_user == 'MC':
        mc_mid_pred, min_values_mc, max_values_mc=utils.mc_model(data=df,sim=10000, n=n, day_type=day_type_user, mark_type=mark_type_user)
        utils.plot_predicted(data=df, mid=mc_mid_pred, lower=min_values_mc, upper=max_values_mc,fcast_dates=fcast_dates,tickerSymbol=tickerSymbol,mark_type=mark_type_user,model_name=model_name_user)
    elif model_name_user == 'ARMA':
        mean_forecast, lower_limits, upper_limits, mu, phi, theta, resid_arima, p_input, q_input=utils.arma_garch_model(data=df,n=n,sim=10000,day_type=day_type_user,mark_type=mark_type_user,arma_or_not=True, optimizer=True, combined=False, vol_type='GARCH', dist_type='normal', mean_type='constant', o_type=0, arma_resid=False)
        utils.plot_predicted(data=df, mid=mean_forecast, lower=lower_limits, upper=upper_limits, fcast_dates=fcast_dates, tickerSymbol=tickerSymbol, mark_type=mark_type_user, model_name=model_name_user)
    elif model_name_user == 'GARCH':
        difference, min_values_ar_garch, max_values_ar_garch, sigma_t, epsilon_t, alpha_params, beta_params, alpha_temp, beta_temp, omega, mu_garch, pred, nu= utils.arma_garch_model(data=df,n=n,sim=10000,day_type=day_type_user,mark_type=mark_type_user,arma_or_not=False, optimizer=False, combined=False, vol_type='GARCH', dist_type='normal', mean_type='constant', o_type=0, arma_resid=False)
        utils.plot_predicted(data=df, mid=difference, lower=min_values_ar_garch, upper=max_values_ar_garch, fcast_dates=fcast_dates, tickerSymbol=tickerSymbol, mark_type=mark_type_user, model_name=model_name_user)
    elif model_name_user == 'ARMA-GARCH':
        difference, max_values_ar_garch, min_values_ar_garch= utils.arma_garch_model(data=df,n=n,sim=10000,day_type=day_type_user,mark_type=mark_type_user,arma_or_not=False, optimizer=False, combined=False, vol_type='GARCH', dist_type='normal', mean_type='constant', o_type=0, arma_resid=False)
        utils.plot_predicted(data=df, mid=difference, lower=min_values_ar_garch, upper=max_values_ar_garch, fcast_dates=fcast_dates, tickerSymbol=tickerSymbol, mark_type=mark_type_user, model_name=model_name_user)
    elif model_name_user == 'LSTM':
        pass
    elif model_name_user =='Hybrid-LSTM':
        pass
    else:
        print('Not valid model')

if __name__ == "__main__":
    main()
