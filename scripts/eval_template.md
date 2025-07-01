# FNO evaluation for decomposition D{iDec} on {device}

- slices : {slices}

Averaged L_2 relative error over time :

- [error plot over time]({errorPlot}) (dashed lines are the errors for an identity operator)

{errors}

Average Inference time for single timestep (s): {avg_inferenceTime}
Inference time after ({tSteps} x {dt})(s): {inferenceTime}

Contour plots :

- [model output]({contourPlotSol})
- [model update]({contourPlotUpdate})
- [absolute error]({contourPlotErr})
- [dedalus output]({contourPlotSolRef})
- [dedalus update]({contourPlotUpdateRef})

Averaged spectrum :

- [full]({spectrumPlot})
- [high frequencies]({spectrumPlotHF})
